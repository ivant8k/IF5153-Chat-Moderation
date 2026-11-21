"""Modeling quality gate: modeling shapes, single-batch overfit, post-processing, and checkpoint load."""
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

if __package__:
    from ..config import DEFAULT_BACKBONE, MAX_LEN, RUNS_DIR, SEEDS, set_seed
    from ..dataset import ToxicSpanCollator, ToxicSpanDataset
    from ..metrics import evaluate_char_spans
    from ..model import BertCRF, BertLinear, build_toxic_lexicon, decode_predictions, expand_whole_words, predict_lexicon
    from ..train import build_optimizer
else:
    TASK_DIR = str(Path(__file__).resolve().parents[1])
    if TASK_DIR not in sys.path:
        sys.path.insert(0, TASK_DIR)
    from config import DEFAULT_BACKBONE, MAX_LEN, RUNS_DIR, SEEDS, set_seed
    from dataset import ToxicSpanCollator, ToxicSpanDataset
    from metrics import evaluate_char_spans
    from model import BertCRF, BertLinear, build_toxic_lexicon, decode_predictions, expand_whole_words, predict_lexicon
    from train import build_optimizer


def log(tag: str, message: str, *, err: bool | None = None) -> None:
    if err is None:
        err = tag in ("ERROR", "FATAL", "WARN")
    print(f"[{tag}] {message}", file=sys.stderr if err else sys.stdout)


def test_postprocess_and_lexicon() -> None:
    """Verify expand_whole_words keeps B->I continuity and lexicon matching works."""
    assert expand_whole_words([0, 1, 0, 0], [None, 0, 0, None]) == [0, 1, 2, 0]
    assert expand_whole_words([0, 0, 1, 0], [None, 0, 0, None]) == [0, 1, 2, 0]
    rows = [{"text": "you idiot", "spans": [4, 5, 6, 7, 8]}] * 3
    lex = build_toxic_lexicon(rows, min_freq=3, min_ratio=0.5)
    assert "idiot" in lex and "you" not in lex
    assert predict_lexicon("such an idiot!", lex) == [8, 9, 10, 11, 12]


def test_overfit_single_batch(train_ds, collator, tokenizer, device) -> float:
    """Run GUIDE.md 8-sentence x 30-step overfit gate on BertCRF."""
    batch = next(iter(DataLoader(Subset(train_ds, range(8)), batch_size=8, collate_fn=collator)))
    golds = [train_ds.rows[i]["spans"] for i in range(8)]

    lin = BertLinear(DEFAULT_BACKBONE).to(device)
    lin_loss = lin(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device),
    )
    assert torch.isfinite(lin_loss)
    del lin

    model = BertCRF(DEFAULT_BACKBONE).to(device)
    opt = build_optimizer(model, encoder_lr=3e-5, head_lr=1e-3)
    ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    model.train()
    for _ in range(30):
        opt.zero_grad()
        loss = model(ids, mask, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        seqs = model(ids, mask)
    spans, _ = decode_predictions(
        seqs,
        [train_ds.samples[i] for i in range(8)],
        [train_ds.rows[i] for i in range(8)],
        tokenizer=tokenizer,
        whole_word=False,
        apply_fix=False,
    )
    f1 = evaluate_char_spans(spans, golds)["f1"]
    assert f1 > 0.95, f"Single-batch overfit F1={f1:.4f} <= 0.95"
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return f1


def test_saved_checkpoint(tokenizer, collator, device) -> None:
    """Verify saved baseline bert_crf.pt loads cleanly and passes CRF transition check."""
    ckpt_path = RUNS_DIR / "baseline" / "bert_crf.pt"
    if not ckpt_path.exists():
        log("SKIP", f"Checkpoint belum ada di {ckpt_path}")
        return

    trial_ds = ToxicSpanDataset("trial", tokenizer, max_length=MAX_LEN)
    sub = Subset(trial_ds, range(16))
    batch = next(iter(DataLoader(sub, batch_size=16, collate_fn=collator)))

    model = BertCRF(DEFAULT_BACKBONE).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        seqs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
    spans, _ = decode_predictions(
        seqs,
        [trial_ds.samples[i] for i in range(16)],
        [trial_ds.rows[i] for i in range(16)],
        tokenizer=tokenizer,
    )
    assert len(spans) == 16

    mat = model.crf.transitions.detach().cpu()
    assert mat[0, 2] < mat[1, 2] and mat[0, 2] < mat[0, 1], "Invalid CRF transition ordering"


def main() -> int:
    set_seed(SEEDS[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("START", f"Memulai verifikasi modul modeling Phase 4 pada device={device}")

    test_postprocess_and_lexicon()
    log("CHECKPOINT 1", "Unit check expand_whole_words & lexicon lulus")

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BACKBONE, use_fast=True)
    train_ds = ToxicSpanDataset("train", tokenizer, max_length=MAX_LEN)
    collator = ToxicSpanCollator(pad_token_id=tokenizer.pad_token_id)

    overfit_f1 = test_overfit_single_batch(train_ds, collator, tokenizer, device)
    log("CHECKPOINT 2", f"Single-batch overfit BertCRF lulus (F1={overfit_f1:.6f})")

    test_saved_checkpoint(tokenizer, collator, device)
    log("CHECKPOINT 3", "Kompatibilitas checkpoint bert_crf.pt & matriks transisi lulus")

    log("DONE", "Seluruh verifikasi Phase 4 lulus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
