"""Data preparation and preprocessing quality gate verification: round-trip F1 and CRF-compatible tensor checks."""
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchcrf import CRF
from transformers import AutoTokenizer

if __package__:
    from ..config import DEFAULT_BACKBONE, ID2LABEL, MAX_LEN, NUM_TAGS, set_seed
    from ..dataset import ToxicSpanCollator, ToxicSpanDataset, bio_to_char_spans, load_semeval_split
    from ..metrics import evaluate_bio_tags, evaluate_char_spans
else:
    TASK_DIR = str(Path(__file__).resolve().parents[1])
    if TASK_DIR not in sys.path:
        sys.path.insert(0, TASK_DIR)
    from config import DEFAULT_BACKBONE, ID2LABEL, MAX_LEN, NUM_TAGS, set_seed
    from dataset import ToxicSpanCollator, ToxicSpanDataset, bio_to_char_spans, load_semeval_split
    from metrics import evaluate_bio_tags, evaluate_char_spans


def log(tag: str, message: str, *, err: bool | None = None) -> None:
    """Emit tagged status line to stdout or stderr."""
    if err is None:
        err = tag in ("ERROR", "FATAL", "WARN")
    print(f"[{tag}] {message}", file=sys.stderr if err else sys.stdout)


def print_table(title: str, rows: list[list], line_width: int = 49) -> None:
    """Render plain aligned table without decorative borders."""
    print(title)
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    total = sum(widths) + 3 * (len(widths) - 1)
    print("-" * max(line_width, total))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())


def run_round_trip_check(split: str, tokenizer) -> dict:
    """Encode split to BIO and decode back to char spans, evaluating SemEval F1."""
    ds = ToxicSpanDataset(split, tokenizer, max_length=MAX_LEN)
    reconstructed = [
        bio_to_char_spans(sample["labels"], sample["offset_mapping"], row["text"])
        for row, sample in zip(ds.rows, ds.samples)
    ]
    golds = [row["spans"] for row in ds.rows]
    stats = evaluate_char_spans(reconstructed, golds)
    stats["dataset"] = ds
    return stats


def run_batch_check(ds: ToxicSpanDataset, tokenizer) -> list[list[str]]:
    """Verify DataLoader batch dtypes, shapes, CRF compatibility, and dynamic padding."""
    collator = ToxicSpanCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collator)
    crf = CRF(num_tags=NUM_TAGS, batch_first=True)

    seq_lengths = set()
    first_shape = None
    for idx, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch["labels"]

        assert input_ids.dtype == torch.long, f"input_ids dtype={input_ids.dtype}"
        assert mask.dtype == torch.bool, f"attention_mask dtype={mask.dtype}"
        assert labels.dtype == torch.long, f"labels dtype={labels.dtype}"
        assert input_ids.ndim == 2 and input_ids.shape == mask.shape == labels.shape
        assert input_ids.shape[1] <= MAX_LEN
        assert bool(mask[:, 0].all()), "mask[:, 0] must be True for pytorch-crf"

        unique_labels = set(labels.unique().tolist())
        assert unique_labels.issubset({0, 1, 2}), f"Unexpected label IDs: {unique_labels}"

        emissions = torch.zeros((*labels.shape, NUM_TAGS), dtype=torch.float)
        nll = -crf(emissions, labels, mask=mask, reduction="mean")
        assert torch.isfinite(nll), f"Non-finite CRF loss: {nll}"

        seq_lengths.add(input_ids.shape[1])
        if idx == 0:
            first_shape = tuple(input_ids.shape)

    assert len(seq_lengths) > 1, "Dynamic padding expected varying batch lengths"
    return [
        ["Split", ds.split],
        ["First batch shape", str(first_shape)],
        ["Distinct batch lengths", str(len(seq_lengths))],
        ["Max batch length observed", str(max(seq_lengths))],
        ["CRF forward check", "PASS"],
    ]


def assert_test_gold_immutable() -> None:
    """Verify fix_spans cannot be applied to tsd_test gold spans."""
    try:
        load_semeval_split("test", apply_fix_spans=True)
    except ValueError:
        return
    raise AssertionError("Expected ValueError when apply_fix_spans=True on test split")


def assert_bpe_leading_space_trim() -> None:
    """Verify bio_to_char_spans trims leading spaces from BPE/SentencePiece-like offsets."""
    text = "You are an idiot!"
    offsets = [(0, 0), (0, 3), (3, 7), (7, 10), (10, 16), (16, 17), (0, 0)]
    tags = ["O", "O", "O", "O", "B-TOXIC", "O", "O"]
    assert bio_to_char_spans(tags, offsets, text) == [11, 12, 13, 14, 15]


def main() -> int:
    set_seed()
    log("START", f"Memulai verifikasi Phase 3 dengan backbone {DEFAULT_BACKBONE}")

    assert_test_gold_immutable()
    assert_bpe_leading_space_trim()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE, use_fast=True)
    assert tokenizer.is_fast, "Tokenizer must be fast"
    log("CHECKPOINT 1", "Tokenizer termuat dan guard tsd_test terverifikasi")

    train_stats = run_round_trip_check("train", tokenizer)
    trial_stats = run_round_trip_check("trial", tokenizer)
    log("CHECKPOINT 2", "Round-trip Char <-> BIO selesai pada train dan trial")

    test_ds = ToxicSpanDataset("test", tokenizer, max_length=MAX_LEN)
    assert len(test_ds) == 2000, f"Unexpected test size: {len(test_ds)}"
    batch_rows = run_batch_check(trial_stats["dataset"], tokenizer)
    _ = run_batch_check(test_ds, tokenizer)

    sample_tags = [[ID2LABEL[t] for t in s["labels"]] for s in trial_stats["dataset"].samples[:10]]
    seq_check = evaluate_bio_tags(sample_tags, sample_tags)
    assert seq_check["f1"] == 1.0 or not any(t != "O" for seq in sample_tags for t in seq)
    log("CHECKPOINT 3", "Verifikasi tensor, dynamic padding, seqeval, dan pytorch-crf lulus")

    print_table(
        "Ringkasan Round-Trip F1 (Phase 3)",
        [
            ["Split", "N", "Overall F1", "SEM", "Empty F1", "Non-Empty F1"],
            [
                "train",
                train_stats["n"],
                f"{train_stats['f1']:.6f}",
                f"{train_stats['sem']:.6f}",
                f"{train_stats['empty_f1']:.6f}",
                f"{train_stats['non_empty_f1']:.6f}",
            ],
            [
                "trial",
                trial_stats["n"],
                f"{trial_stats['f1']:.6f}",
                f"{trial_stats['sem']:.6f}",
                f"{trial_stats['empty_f1']:.6f}",
                f"{trial_stats['non_empty_f1']:.6f}",
            ],
        ],
    )
    print()
    print_table("Ringkasan Batch & Tensor Check", batch_rows)

    if trial_stats["f1"] < 0.96:
        log("ERROR", f"Round-trip F1 pada tsd_trial ({trial_stats['f1']:.6f}) < 0.96")
        return 1

    log("DONE", "Seluruh verifikasi Phase 3 lulus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
