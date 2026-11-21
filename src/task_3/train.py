"""Task 3 training pipeline for Lexicon, BertLinear, and BertCRF."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

if __package__:
    from .config import DEFAULT_BACKBONE, ID2LABEL, MAX_LEN, RESULTS_DIR, RUNS_DIR, SEEDS, set_seed
    from .dataset import ToxicSpanCollator, ToxicSpanDataset
    from .metrics import evaluate_bio_tags, evaluate_char_spans
    from .model import BertCRF, BertLinear, build_toxic_lexicon, decode_predictions, predict_lexicon
else:
    TASK_DIR = str(Path(__file__).resolve().parent)
    if TASK_DIR not in sys.path:
        sys.path.insert(0, TASK_DIR)
    from config import DEFAULT_BACKBONE, ID2LABEL, MAX_LEN, RESULTS_DIR, RUNS_DIR, SEEDS, set_seed
    from dataset import ToxicSpanCollator, ToxicSpanDataset
    from metrics import evaluate_bio_tags, evaluate_char_spans
    from model import BertCRF, BertLinear, build_toxic_lexicon, decode_predictions, predict_lexicon


def log(tag: str, message: str, *, err: bool | None = None) -> None:
    if err is None:
        err = tag in ("ERROR", "FATAL", "WARN")
    print(f"[{tag}] {message}", file=sys.stderr if err else sys.stdout, flush=True)


def print_table(title: str, rows: list[list], line_width: int = 49) -> None:
    print(title)
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    total = sum(widths) + 3 * (len(widths) - 1)
    print("-" * max(line_width, total))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())


def build_optimizer(model: torch.nn.Module, encoder_lr: float, head_lr: float) -> torch.optim.Optimizer:
    groups = [
        {"params": model.encoder.parameters(), "lr": encoder_lr, "weight_decay": 0.01},
        {"params": model.classifier.parameters(), "lr": head_lr, "weight_decay": 0.01},
    ]
    if hasattr(model, "crf"):
        groups.append({"params": model.crf.parameters(), "lr": head_lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups)


def evaluate_neural(
    model: torch.nn.Module,
    loader: DataLoader,
    samples: list[dict],
    rows: list[dict],
    word_ids_list: list[list[int | None]],
    device: torch.device,
) -> tuple[dict, dict, list[list[int]], list[list[str]]]:
    model.eval()
    pred_seqs = []
    with torch.no_grad():
        for batch in loader:
            pred_seqs.extend(model(batch["input_ids"].to(device), batch["attention_mask"].to(device)))

    pred_spans, pred_tags = decode_predictions(
        pred_seqs,
        samples,
        rows,
        word_ids_list=word_ids_list,
        whole_word=True,
        apply_fix=True,
    )
    char_stats = evaluate_char_spans(pred_spans, [r["spans"] for r in rows])
    gold_tags = [[ID2LABEL[t] for t in s["labels"]] for s in samples]
    seq_stats = evaluate_bio_tags(pred_tags, gold_tags)
    return char_stats, seq_stats, pred_spans, pred_tags


def measure_latency_ms(
    model: torch.nn.Module,
    ds: ToxicSpanDataset | Subset,
    collator: ToxicSpanCollator,
    device: torch.device,
    n_warmup: int = 20,
) -> float:
    model.eval()
    items = list(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator))
    with torch.no_grad():
        for batch in items[:n_warmup]:
            _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for batch in items:
            _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / max(len(items), 1)


def train_neural(args, model_key: str, tokenizer, train_ds, trial_ds, trial_wids, collator, device) -> dict:
    set_seed(args.seed)
    model_cls = BertCRF if model_key == "crf" else BertLinear
    run_name = f"bert_{model_key}"

    tr_subset = Subset(train_ds, range(min(64, len(train_ds)))) if args.smoke else train_ds
    va_subset = Subset(trial_ds, range(min(16, len(trial_ds)))) if args.smoke else trial_ds
    va_samples = [trial_ds.samples[i] for i in range(len(va_subset))]
    va_rows = [trial_ds.rows[i] for i in range(len(va_subset))]
    va_wids = trial_wids[:len(va_subset)]

    tr_loader = DataLoader(tr_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    va_loader = DataLoader(va_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    model = model_cls(args.backbone).to(device)
    optimizer = build_optimizer(model, args.encoder_lr, args.head_lr)
    total_steps = max(len(tr_loader) * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    ckpt_path = args.run_dir / f"{run_name}.pt"
    best_f1 = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        total_loss = 0.0
        for batch in tr_loader:
            optimizer.zero_grad()
            loss = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["labels"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(tr_loader), 1)
        char_stats, seq_stats, _, _ = evaluate_neural(model, va_loader, va_samples, va_rows, va_wids, device)
        dt = time.perf_counter() - t_ep
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "trial_char_f1": char_stats["f1"],
            "trial_empty_f1": char_stats["empty_f1"],
            "trial_non_empty_f1": char_stats["non_empty_f1"],
            "trial_seq_f1": seq_stats["f1"],
            "sec": dt,
        })
        log(
            f"CHECKPOINT {epoch}",
            f"{run_name} ep {epoch}/{args.epochs} | loss={avg_loss:.4f} | trial_f1={char_stats['f1']:.5f} | seq_f1={seq_stats['f1']:.5f} ({dt:.1f}s)",
        )
        if char_stats["f1"] > best_f1:
            best_f1 = char_stats["f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    char_stats, seq_stats, pred_spans, pred_tags = evaluate_neural(
        model, va_loader, va_samples, va_rows, va_wids, device
    )
    latency_ms = measure_latency_ms(model, va_subset, collator, device)

    (args.out_dir / f"{run_name}_trial_preds.json").write_text(
        json.dumps({"spans": pred_spans, "bio_tags": pred_tags}),
        encoding="utf-8",
    )

    del model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "best_epoch": best_epoch,
        "char": char_stats,
        "seqeval": seq_stats,
        "latency_ms": latency_ms,
        "history": history,
        "ckpt_path": str(ckpt_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Task 3 toxic span extraction models.")
    parser.add_argument("--model", choices=("crf", "linear", "lexicon", "all"), default="crf")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--run-dir", type=Path, default=RUNS_DIR / "baseline")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "baseline")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.run_dir = args.run_dir / "smoke"
        args.out_dir = args.out_dir / "smoke"

    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("START", f"Training model={args.model} backbone={args.backbone} device={device} smoke={args.smoke}")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    train_ds = ToxicSpanDataset("train", tokenizer, max_length=MAX_LEN)
    trial_ds = ToxicSpanDataset("trial", tokenizer, max_length=MAX_LEN)
    collator = ToxicSpanCollator(pad_token_id=tokenizer.pad_token_id)
    trial_wids = [
        tokenizer(r["text"], truncation=True, max_length=MAX_LEN).word_ids(0)
        for r in trial_ds.rows
    ]

    targets = ["lexicon", "linear", "crf"] if args.model == "all" else [args.model]
    results = {}
    table_rows = [["Model", "BestEp", "Char F1", "Empty F1", "NonEmpty F1", "Seqeval F1", "Latency(ms)"]]

    for key in targets:
        if key == "lexicon":
            tr_rows = train_ds.rows[:64] if args.smoke else train_ds.rows
            va_rows = trial_ds.rows[:16] if args.smoke else trial_ds.rows
            lexicon = build_toxic_lexicon(tr_rows)
            t0 = time.perf_counter()
            preds = [predict_lexicon(r["text"], lexicon) for r in va_rows]
            lat_ms = (time.perf_counter() - t0) * 1000.0 / max(len(va_rows), 1)
            c_stat = evaluate_char_spans(preds, [r["spans"] for r in va_rows])
            (args.out_dir / "lexicon_trial_preds.json").write_text(json.dumps(preds), encoding="utf-8")
            results["lexicon"] = {"char": c_stat, "latency_ms": lat_ms, "vocab_size": len(lexicon)}
            table_rows.append([
                "lexicon", "-", f"{c_stat['f1']:.5f}", f"{c_stat['empty_f1']:.5f}",
                f"{c_stat['non_empty_f1']:.5f}", "-", f"{lat_ms:.3f}",
            ])
        else:
            res = train_neural(args, key, tokenizer, train_ds, trial_ds, trial_wids, collator, device)
            results[f"bert_{key}"] = res
            table_rows.append([
                f"bert_{key}", str(res["best_epoch"]), f"{res['char']['f1']:.5f}",
                f"{res['char']['empty_f1']:.5f}", f"{res['char']['non_empty_f1']:.5f}",
                f"{res['seqeval']['f1']:.5f}", f"{res['latency_ms']:.2f}",
            ])

    print_table("Ringkasan Training Phase 4 (tsd_trial)", table_rows)
    (args.out_dir / "train_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    log("DONE", f"Hasil training tersimpan di {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
