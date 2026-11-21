"""Task 3 dataset preparation: span cleaning, BIO conversion, and dynamic padding."""
import ast
from collections import Counter
import csv
import sys
from typing import Sequence

import torch
from torch.utils.data import Dataset

if __package__:
    from .config import DATA_FILES, EVAL_DIR, ID2LABEL, LABEL2ID, MAX_LEN
else:
    from config import DATA_FILES, EVAL_DIR, ID2LABEL, LABEL2ID, MAX_LEN

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from fix_spans import _contiguous_ranges, fix_spans


def load_semeval_split(split: str, apply_fix_spans: bool | None = None) -> list[dict]:
    """Load SemEval-2021 CSV split, applying fix_spans only to train and trial."""
    if split not in DATA_FILES:
        raise ValueError(f"Unknown split '{split}'. Expected one of {tuple(DATA_FILES)}")
    if apply_fix_spans is None:
        apply_fix_spans = split in ("train", "trial")
    if split == "test" and apply_fix_spans:
        raise ValueError("fix_spans must not be applied to tsd_test gold spans")

    rows = []
    with open(DATA_FILES[split], encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = row["text"]
            spans = ast.literal_eval(row["spans"])
            if apply_fix_spans:
                spans = fix_spans(spans, text)
            rows.append({"text": text, "spans": spans})
    return rows


def trim_offset(text: str, start: int, end: int) -> tuple[int, int]:
    """Strip leading/trailing whitespace from a token offset for BPE/SentencePiece compatibility."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def char_spans_to_bio(
    text: str,
    spans: list[int],
    tokenizer,
    max_length: int = MAX_LEN,
) -> dict:
    """Convert character offsets into CRF-safe subword BIO label IDs."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("char_spans_to_bio requires a fast tokenizer (use_fast=True)")

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = encoding["offset_mapping"]
    sequence_ids = encoding.sequence_ids(0)

    ranges = _contiguous_ranges(sorted(set(spans)))
    char_to_range = {}
    for range_idx, (start, end) in enumerate(ranges):
        for pos in range(start, end + 1):
            char_to_range[pos] = range_idx

    labels = []
    prev_range_idx = None
    for idx, (orig_start, orig_end) in enumerate(offsets):
        if sequence_ids[idx] is None or orig_start == orig_end:
            labels.append(LABEL2ID["O"])
            prev_range_idx = None
            continue

        tok_start, tok_end = trim_offset(text, orig_start, orig_end)
        if tok_start == tok_end:
            labels.append(LABEL2ID["O"])
            continue

        matched_ranges = [
            char_to_range[pos]
            for pos in range(tok_start, tok_end)
            if pos in char_to_range
        ]
        tok_len = tok_end - tok_start
        if len(matched_ranges) * 2 >= tok_len:
            cur_range_idx = Counter(matched_ranges).most_common(1)[0][0]
            tag_name = "I-TOXIC" if prev_range_idx == cur_range_idx else "B-TOXIC"
            labels.append(LABEL2ID[tag_name])
            prev_range_idx = cur_range_idx
        else:
            labels.append(LABEL2ID["O"])
            prev_range_idx = None

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "offset_mapping": offsets,
    }


def bio_to_char_spans(
    tags: Sequence[int | str],
    offset_mapping: Sequence[tuple[int, int]],
    text: str,
) -> list[int]:
    """Reconstruct sorted character offsets from BIO tags with whitespace gap bridging."""
    str_tags = [ID2LABEL[t] if isinstance(t, int) else t for t in tags]
    spans = []
    prev_toxic_end = None

    for tag, (orig_start, orig_end) in zip(str_tags, offset_mapping):
        if orig_start == orig_end:
            prev_toxic_end = None
            continue

        tok_start, tok_end = trim_offset(text, orig_start, orig_end)
        if tok_start == tok_end:
            continue
        if tag == "O":
            prev_toxic_end = None
            continue

        if (
            tag == "I-TOXIC"
            and prev_toxic_end is not None
            and tok_start > prev_toxic_end
            and text[prev_toxic_end:tok_start].isspace()
        ):
            spans.extend(range(prev_toxic_end, tok_start))

        spans.extend(range(tok_start, tok_end))
        prev_toxic_end = tok_end

    return sorted(set(spans))


class ToxicSpanDataset(Dataset):
    """PyTorch Dataset encoding SemEval-2021 rows into subword BIO sequences."""

    def __init__(
        self,
        split: str,
        tokenizer,
        max_length: int = MAX_LEN,
        apply_fix_spans: bool | None = None,
    ) -> None:
        self.split = split
        self.rows = load_semeval_split(split, apply_fix_spans=apply_fix_spans)
        self.samples = [
            char_spans_to_bio(r["text"], r["spans"], tokenizer, max_length=max_length)
            for r in self.rows
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class ToxicSpanCollator:
    """Dynamic-padding collator producing CRF-compatible tensors (no -100, bool mask)."""

    def __init__(self, pad_token_id: int, pad_label_id: int = LABEL2ID["O"]) -> None:
        self.pad_token_id = pad_token_id if pad_token_id is not None else 0
        self.pad_label_id = pad_label_id

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        labels = []
        offset_mapping = []

        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([bool(m) for m in item["attention_mask"]] + [False] * pad_len)
            labels.append(item["labels"] + [self.pad_label_id] * pad_len)
            if "offset_mapping" in item:
                offset_mapping.append(item["offset_mapping"])

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        if offset_mapping:
            out["offset_mapping"] = offset_mapping
        return out
