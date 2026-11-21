"""Task 3 models, lexicon baseline, and post-processing helpers."""
from collections import Counter
import re
import sys
from typing import Sequence

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoConfig, AutoModel

if __package__:
    from .config import EVAL_DIR, ID2LABEL, MAX_LEN, NUM_TAGS
    from .dataset import bio_to_char_spans
else:
    from config import EVAL_DIR, ID2LABEL, MAX_LEN, NUM_TAGS
    from dataset import bio_to_char_spans

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from fix_spans import fix_spans

WORD_RE = re.compile(r"\b\w+\b")


class _BertTokenBase(nn.Module):
    """Shared Transformer encoder + dropout + linear emission head."""

    def __init__(self, model_name: str, num_tags: int = NUM_TAGS, dropout: float = 0.2) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_tags)

    def encode_emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask.long())
        return self.classifier(self.dropout(outputs.last_hidden_state))


class BertLinear(_BertTokenBase):
    """Pretrained Transformer with linear token-classification head (no CRF)."""

    def __init__(self, model_name: str, num_tags: int = NUM_TAGS, dropout: float = 0.2) -> None:
        super().__init__(model_name, num_tags=num_tags, dropout=dropout)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | list[list[int]]:
        emissions = self.encode_emissions(input_ids, attention_mask)
        if labels is not None:
            masked_labels = labels.masked_fill(~attention_mask, -100)
            return self.loss_fn(emissions.view(-1, emissions.size(-1)), masked_labels.view(-1))
        pred_ids = emissions.argmax(dim=-1)
        lengths = attention_mask.sum(dim=1).tolist()
        return [seq[:length].tolist() for seq, length in zip(pred_ids, lengths)]


class BertCRF(_BertTokenBase):
    """Pretrained Transformer encoder with Linear-Chain CRF head."""

    def __init__(self, model_name: str, num_tags: int = NUM_TAGS, dropout: float = 0.2) -> None:
        super().__init__(model_name, num_tags=num_tags, dropout=dropout)
        self.crf = CRF(num_tags=num_tags, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | list[list[int]]:
        emissions = self.encode_emissions(input_ids, attention_mask)
        if labels is not None:
            return -self.crf(emissions, labels, mask=attention_mask, reduction="mean")
        return self.crf.decode(emissions, mask=attention_mask)


def build_toxic_lexicon(rows: Sequence[dict], min_freq: int = 3, min_ratio: float = 0.5) -> set[str]:
    """Extract high-toxicity-ratio word vocabulary from training rows."""
    total_cnt: Counter[str] = Counter()
    toxic_cnt: Counter[str] = Counter()
    for row in rows:
        span_set = set(row["spans"])
        for m in WORD_RE.finditer(row["text"]):
            word = m.group(0).lower()
            total_cnt[word] += 1
            tok_len = m.end() - m.start()
            overlap = sum(pos in span_set for pos in range(m.start(), m.end()))
            if overlap * 2 >= tok_len:
                toxic_cnt[word] += 1
    return {
        word
        for word, tc in toxic_cnt.items()
        if tc >= min_freq and (tc / total_cnt[word]) >= min_ratio
    }


def predict_lexicon(text: str, lexicon: set[str]) -> list[int]:
    """Return sorted character offsets for words matching the toxic lexicon."""
    spans = []
    for m in WORD_RE.finditer(text):
        if m.group(0).lower() in lexicon:
            spans.extend(range(m.start(), m.end()))
    return spans


def expand_whole_words(pred_seq: Sequence[int], word_ids: Sequence[int | None]) -> list[int]:
    """Propagate partial subword toxic predictions across the entire parent word."""
    toxic_word_ids = {
        wid
        for tag_id, wid in zip(pred_seq, word_ids)
        if wid is not None and tag_id != 0
    }
    if not toxic_word_ids:
        return list(pred_seq)

    out = list(pred_seq)
    for idx, wid in enumerate(word_ids):
        if wid in toxic_word_ids:
            if out[idx] == 0:
                prev_toxic = idx > 0 and out[idx - 1] != 0
                out[idx] = 2 if prev_toxic else 1
            elif idx > 0 and word_ids[idx - 1] == wid and out[idx - 1] != 0:
                out[idx] = 2
    return out


def decode_predictions(
    pred_seqs: Sequence[Sequence[int]],
    samples: Sequence[dict],
    rows: Sequence[dict],
    tokenizer=None,
    word_ids_list: Sequence[Sequence[int | None]] | None = None,
    whole_word: bool = True,
    apply_fix: bool = True,
    max_length: int = MAX_LEN,
) -> tuple[list[list[int]], list[list[str]]]:
    """Post-process predicted tag IDs into character spans and BIO string tags."""
    if whole_word and word_ids_list is None:
        if tokenizer is None:
            raise ValueError("tokenizer or word_ids_list is required when whole_word=True")
        word_ids_list = [
            tokenizer(r["text"], truncation=True, max_length=max_length).word_ids(0)
            for r in rows
        ]

    final_seqs = (
        [expand_whole_words(seq, wids) for seq, wids in zip(pred_seqs, word_ids_list)]
        if whole_word
        else [list(seq) for seq in pred_seqs]
    )

    pred_spans = [
        bio_to_char_spans(seq, sample["offset_mapping"], row["text"])
        for seq, sample, row in zip(final_seqs, samples, rows)
    ]
    if apply_fix:
        pred_spans = [fix_spans(sp, row["text"]) for sp, row in zip(pred_spans, rows)]

    pred_tags = [[ID2LABEL[t] for t in seq] for seq in final_seqs]
    return pred_spans, pred_tags
