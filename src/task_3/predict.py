"""End-to-end Task 3 inference wrapper combining BertCRF with RC-03 smart masking."""
import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

if __package__:
    from .censor import align_and_mask, task1_tokenize
    from .config import DEFAULT_BACKBONE, ID2LABEL, MAX_LEN, RUNS_DIR
    from .model import BertCRF
else:
    TASK_DIR = str(Path(__file__).resolve().parent)
    if TASK_DIR not in sys.path:
        sys.path.insert(0, TASK_DIR)
    from censor import align_and_mask, task1_tokenize
    from config import DEFAULT_BACKBONE, ID2LABEL, MAX_LEN, RUNS_DIR
    from model import BertCRF


class ToxicSpanCensor:
    """Word-level BertCRF predictor paired with RC-03 character-span masking."""

    def __init__(
        self,
        ckpt_path: Path = RUNS_DIR / "baseline" / "bert_crf.pt",
        backbone: str = DEFAULT_BACKBONE,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True)
        self.model = BertCRF(backbone).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def predict_word_tags(self, words: list[str]) -> list[str]:
        """Predict BIO tags for pre-tokenized words using subword-to-word aggregation."""
        if not words:
            return []

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(0)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].bool().to(self.device)

        with torch.no_grad():
            subword_tag_ids = self.model(input_ids, attention_mask)[0]

        word_subtags: dict[int, list[int]] = {}
        for wid, tag_id in zip(word_ids, subword_tag_ids):
            if wid is not None:
                word_subtags.setdefault(wid, []).append(tag_id)

        out_tags: list[str] = []
        for wid in range(len(words)):
            tags = word_subtags.get(wid, [])
            if any(t != 0 for t in tags):
                prev_toxic = bool(out_tags and out_tags[-1] != "O")
                first_nonzero = next(t for t in tags if t != 0)
                if first_nonzero == 2 and prev_toxic:
                    out_tags.append("I-TOXIC")
                else:
                    out_tags.append(ID2LABEL[2] if prev_toxic and tags[0] == 2 else "B-TOXIC")
            else:
                out_tags.append("O")
        return out_tags

    def censor(
        self,
        raw_text: str,
        delim_norm_text: str | None = None,
        is_toxic_trigger: bool = True,
        raw_slots: list[tuple[str, int, int]] | None = None,
        mask_char: str = "*",
    ) -> dict:
        """Censor raw_text using Task 1 slots/normalized text and Task 2 trigger."""
        if raw_slots is None:
            raw_slots = task1_tokenize(raw_text)
        if delim_norm_text is None:
            delim_norm_text = " | ".join(tok for tok, _, _ in raw_slots)
        return align_and_mask(
            raw_text=raw_text,
            delim_norm_text=delim_norm_text,
            is_toxic_trigger=is_toxic_trigger,
            predict_word_tags_fn=self.predict_word_tags,
            raw_slots=raw_slots,
            mask_char=mask_char,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 3 end-to-end toxic span censorship.")
    parser.add_argument("--raw-text", required=True)
    parser.add_argument("--delim-norm", default=None)
    parser.add_argument("--non-toxic", action="store_true", help="Simulate Task 2 is_toxic_trigger=False.")
    parser.add_argument("--ckpt", type=Path, default=RUNS_DIR / "baseline" / "bert_crf.pt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    censor_pipe = ToxicSpanCensor(ckpt_path=args.ckpt)
    result = censor_pipe.censor(
        raw_text=args.raw_text,
        delim_norm_text=args.delim_norm,
        is_toxic_trigger=not args.non_toxic,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
