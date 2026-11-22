"""Phase 5 intrinsic evaluation on tsd_trial & tsd_test and qualitative error extraction."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import sem
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

if __package__:
    from .config import DEFAULT_BACKBONE, EVAL_DIR, ID2LABEL, MAX_LEN, RESULTS_DIR, RUNS_DIR, SEEDS, set_seed
    from .dataset import ToxicSpanCollator, ToxicSpanDataset, load_semeval_split
    from .metrics import evaluate_bio_tags, evaluate_char_spans
    from .model import BertCRF, BertLinear, build_toxic_lexicon, decode_predictions, predict_lexicon
    from .train import measure_latency_ms
else:
    TASK_DIR = str(Path(__file__).resolve().parent)
    if TASK_DIR not in sys.path:
        sys.path.insert(0, TASK_DIR)
    from config import DEFAULT_BACKBONE, EVAL_DIR, ID2LABEL, MAX_LEN, RESULTS_DIR, RUNS_DIR, SEEDS, set_seed
    from dataset import ToxicSpanCollator, ToxicSpanDataset, load_semeval_split
    from metrics import evaluate_bio_tags, evaluate_char_spans
    from model import BertCRF, BertLinear, build_toxic_lexicon, decode_predictions, predict_lexicon
    from train import measure_latency_ms

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from fix_spans import _contiguous_ranges
from semeval2021 import f1 as semeval_f1


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


def split_gold_subset(gold_spans: list[int], text: str) -> tuple[str, bool]:
    """Classify sample by contiguous gold span count and presence of >=4-word span."""
    if not gold_spans:
        return "empty", False
    ranges = _contiguous_ranges(sorted(set(gold_spans)))
    is_long = any(len(text[b : e + 1].split()) >= 4 for b, e in ranges)
    return ("single" if len(ranges) == 1 else "multi"), is_long


def _mean_sem(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "f1": float(np.mean(values)) if values else 0.0,
        "sem": float(sem(values)) if len(values) > 1 else 0.0,
    }


def summarize_predictions(
    preds: list[list[int]],
    golds: list[list[int]],
    rows: list[dict],
    pred_tags: list[list[str]] | None = None,
    gold_tags: list[list[str]] | None = None,
) -> tuple[dict, list[float]]:
    """Compute overall Char F1, subset breakdowns, and optional seqeval metrics."""
    overall = evaluate_char_spans(preds, golds)
    row_scores = [semeval_f1(p, g) for p, g in zip(preds, golds)]

    buckets: dict[str, list[float]] = {"empty": [], "single": [], "multi": [], "long_phrase": []}
    for score, gold, row in zip(row_scores, golds, rows):
        subset, is_long = split_gold_subset(gold, row["text"])
        buckets[subset].append(score)
        if is_long:
            buckets["long_phrase"].append(score)

    out = {
        "char": overall,
        "subsets": {k: _mean_sem(v) for k, v in buckets.items()},
    }
    if pred_tags is not None and gold_tags is not None:
        out["seqeval"] = evaluate_bio_tags(pred_tags, gold_tags)
    return out, row_scores


def classify_error(pred: list[int], gold: list[int], score: float) -> str:
    """Categorize prediction error type for qualitative error analysis."""
    if score == 1.0:
        return "exact_match"
    if not gold and pred:
        return "empty_false_positive"
    if gold and not pred:
        return "missed_toxic"
    if score == 0.0:
        return "zero_overlap_mismatch"
    if len(pred) > len(gold):
        return "partial_over_extension"
    if len(pred) < len(gold):
        return "partial_under_extension"
    return "partial_shifted"


def extract_span_texts(spans: list[int], text: str) -> list[str]:
    """Slice contiguous span strings from text for readable error inspection."""
    if not spans:
        return []
    return [text[b : e + 1] for b, e in _contiguous_ranges(sorted(set(spans)))]


def build_error_report(
    rows: list[dict],
    crf_preds: list[list[int]],
    crf_scores: list[float],
    lin_preds: list[list[int]],
    lin_scores: list[float],
) -> dict:
    """Summarize error categories and pick representative worst/comparison samples."""
    cat_counts: dict[str, int] = {}
    records = []
    for idx, (row, cp, cs, lp, ls) in enumerate(zip(rows, crf_preds, crf_scores, lin_preds, lin_scores)):
        err_type = classify_error(cp, row["spans"], cs)
        cat_counts[err_type] = cat_counts.get(err_type, 0) + 1
        records.append({
            "idx": idx,
            "text": row["text"],
            "gold_texts": extract_span_texts(row["spans"], row["text"]),
            "crf_texts": extract_span_texts(cp, row["text"]),
            "linear_texts": extract_span_texts(lp, row["text"]),
            "crf_f1": round(float(cs), 4),
            "linear_f1": round(float(ls), 4),
            "error_type": err_type,
        })

    zero_cases = [r for r in records if r["crf_f1"] == 0.0][:15]
    partial_cases = sorted(
        [r for r in records if 0.0 < r["crf_f1"] < 0.5],
        key=lambda r: r["crf_f1"],
    )[:15]
    crf_wins = sorted(
        [r for r in records if r["crf_f1"] - r["linear_f1"] >= 0.25],
        key=lambda r: r["crf_f1"] - r["linear_f1"],
        reverse=True,
    )[:10]
    lin_wins = sorted(
        [r for r in records if r["linear_f1"] - r["crf_f1"] >= 0.25],
        key=lambda r: r["linear_f1"] - r["crf_f1"],
        reverse=True,
    )[:5]

    return {
        "n": len(rows),
        "error_category_counts": cat_counts,
        "worst_zero_f1": zero_cases,
        "worst_partial_f1": partial_cases,
        "crf_beats_linear": crf_wins,
        "linear_beats_crf": lin_wins,
    }


def predict_neural_seqs(
    model_cls,
    ckpt_path: Path,
    backbone: str,
    ds: ToxicSpanDataset,
    collator: ToxicSpanCollator,
    device: torch.device,
    batch_size: int,
    latency_samples: int,
) -> tuple[list[list[int]], float]:
    """Load checkpoint once, decode full dataset, and measure batch_size=1 latency."""
    model = model_cls(backbone).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    pred_seqs = []
    with torch.no_grad():
        for batch in loader:
            pred_seqs.extend(model(batch["input_ids"].to(device), batch["attention_mask"].to(device)))

    lat_ds = Subset(ds, range(min(latency_samples, len(ds))))
    latency_ms = measure_latency_ms(model, lat_ds, collator, device)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred_seqs, latency_ms


def evaluate_split(
    ds: ToxicSpanDataset,
    lexicon: set[str],
    tokenizer,
    collator: ToxicSpanCollator,
    ckpt_dir: Path,
    backbone: str,
    device: torch.device,
    batch_size: int,
    latency_samples: int,
) -> dict:
    """Evaluate all 4 models on one SemEval split without altering gold spans."""
    rows = ds.rows
    golds = [r["spans"] for r in rows]
    gold_tags = [[ID2LABEL[t] for t in s["labels"]] for s in ds.samples]
    word_ids_list = [
        tokenizer(r["text"], truncation=True, max_length=MAX_LEN).word_ids(0)
        for r in rows
    ]

    t0 = time.perf_counter()
    lex_preds = [predict_lexicon(r["text"], lexicon) for r in rows]
    lex_lat_ms = (time.perf_counter() - t0) * 1000.0 / max(len(rows), 1)
    lex_stat, _ = summarize_predictions(lex_preds, golds, rows)
    lex_stat["latency_ms"] = lex_lat_ms

    lin_seqs, lin_lat_ms = predict_neural_seqs(
        BertLinear, ckpt_dir / "bert_linear.pt", backbone, ds, collator, device, batch_size, latency_samples
    )
    lin_preds, lin_tags = decode_predictions(
        lin_seqs, ds.samples, rows, word_ids_list=word_ids_list, whole_word=False, apply_fix=False
    )
    lin_stat, lin_scores = summarize_predictions(lin_preds, golds, rows, lin_tags, gold_tags)
    lin_stat["latency_ms"] = lin_lat_ms

    crf_seqs, crf_lat_ms = predict_neural_seqs(
        BertCRF, ckpt_dir / "bert_crf.pt", backbone, ds, collator, device, batch_size, latency_samples
    )
    crf_raw_preds, crf_raw_tags = decode_predictions(
        crf_seqs, ds.samples, rows, word_ids_list=word_ids_list, whole_word=False, apply_fix=False
    )
    crf_raw_stat, _ = summarize_predictions(crf_raw_preds, golds, rows, crf_raw_tags, gold_tags)
    crf_raw_stat["latency_ms"] = crf_lat_ms

    crf_post_preds, crf_post_tags = decode_predictions(
        crf_seqs, ds.samples, rows, word_ids_list=word_ids_list, whole_word=True, apply_fix=True
    )
    crf_post_stat, crf_scores = summarize_predictions(crf_post_preds, golds, rows, crf_post_tags, gold_tags)
    crf_post_stat["latency_ms"] = crf_lat_ms

    return {
        "models": {
            "lexicon": lex_stat,
            "bert_linear": lin_stat,
            "bert_crf_raw": crf_raw_stat,
            "bert_crf_post": crf_post_stat,
        },
        "error_report": build_error_report(rows, crf_post_preds, crf_scores, lin_preds, lin_scores),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Task 3 models on tsd_trial and tsd_test.")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--ckpt-dir", type=Path, default=RUNS_DIR / "baseline")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "eval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-samples", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(SEEDS[0])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for req in ("bert_linear.pt", "bert_crf.pt"):
        if not (args.ckpt_dir / req).exists():
            log("ERROR", f"Missing checkpoint: {args.ckpt_dir / req}")
            return 1

    log("START", f"Evaluating checkpoints from {args.ckpt_dir} on device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    collator = ToxicSpanCollator(pad_token_id=tokenizer.pad_token_id)

    train_rows = load_semeval_split("train")
    lexicon = build_toxic_lexicon(train_rows)

    trial_ds = ToxicSpanDataset("trial", tokenizer, max_length=MAX_LEN)
    test_ds = ToxicSpanDataset("test", tokenizer, max_length=MAX_LEN)

    trial_res = evaluate_split(
        trial_ds, lexicon, tokenizer, collator, args.ckpt_dir, args.backbone, device, args.batch_size, args.latency_samples
    )
    log("CHECKPOINT 1", "Evaluasi tsd_trial selesai")

    test_res = evaluate_split(
        test_ds, lexicon, tokenizer, collator, args.ckpt_dir, args.backbone, device, args.batch_size, args.latency_samples
    )
    log("CHECKPOINT 2", "Evaluasi tsd_test selesai")

    summary = {
        "trial": trial_res["models"],
        "test": test_res["models"],
        "test_error_categories": test_res["error_report"]["error_category_counts"],
    }
    (args.out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "error_cases.json").write_text(
        json.dumps({"trial": trial_res["error_report"], "test": test_res["error_report"]}, indent=2),
        encoding="utf-8",
    )

    comp_rows = [["Split", "Model", "Char F1", "SEM", "Seqeval P", "Seqeval R", "Seqeval F1", "Latency(ms)"]]
    for split_name, res in (("trial", trial_res), ("test", test_res)):
        for m_name in ("lexicon", "bert_linear", "bert_crf_raw", "bert_crf_post"):
            st = res["models"][m_name]
            sq = st.get("seqeval")
            comp_rows.append([
                split_name,
                m_name,
                f"{st['char']['f1']:.5f}",
                f"{st['char']['sem']:.5f}",
                f"{sq['precision']:.5f}" if sq else "-",
                f"{sq['recall']:.5f}" if sq else "-",
                f"{sq['f1']:.5f}" if sq else "-",
                f"{st['latency_ms']:.2f}",
            ])
    print_table("Evaluasi Kuantitatif Phase 5 (tsd_trial & tsd_test)", comp_rows)

    sub_rows = [["Split", "Model", "Empty F1 (N)", "Single F1 (N)", "Multi F1 (N)", "Long>=4w F1 (N)"]]
    for split_name, res in (("trial", trial_res), ("test", test_res)):
        for m_name in ("lexicon", "bert_linear", "bert_crf_raw", "bert_crf_post"):
            sb = res["models"][m_name]["subsets"]
            sub_rows.append([
                split_name,
                m_name,
                f"{sb['empty']['f1']:.5f} ({sb['empty']['n']})",
                f"{sb['single']['f1']:.5f} ({sb['single']['n']})",
                f"{sb['multi']['f1']:.5f} ({sb['multi']['n']})",
                f"{sb['long_phrase']['f1']:.5f} ({sb['long_phrase']['n']})",
            ])
    print()
    print_table("Breakdown Subset per Tipe Kalimat", sub_rows)

    err_rows = [["Kategori Error (tsd_test)", "Jumlah", "Persentase"]]
    total_test = test_res["error_report"]["n"]
    for cat, cnt in sorted(test_res["error_report"]["error_category_counts"].items(), key=lambda x: -x[1]):
        err_rows.append([cat, str(cnt), f"{100.0 * cnt / total_test:.2f}%"])
    print()
    print_table("Distribusi Kategori Error (BERT-CRF+Post di tsd_test)", err_rows)

    log("DONE", f"Hasil evaluasi & error cases tersimpan di {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
