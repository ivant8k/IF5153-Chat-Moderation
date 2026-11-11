"""Predict: run -> file output model per set (dev/test, greedy/beam), latensi"""
import argparse
import random

import numpy as np

from common import export_run, load_set, pred_tag, read_json, set_names, to_input, write_json, write_preds
from config import BACKBONES, RUNS, SMOKE_EVAL_ROWS, SMOKE_RUNS
from model import device_info, generate, latency, load_backbone, load_checkpoint, setup_device
from runs import get_run, run_dir


def load_run_model(spec, out_dir, device):
    if spec["kind"] == "zeroshot":
        return load_backbone(spec["backbone"], device)
    return load_checkpoint(out_dir / "checkpoints" / "best", device)


def run_spec(run_id, smoke):
    """config.json run jika ada, selain itu dari daftar run"""
    path = run_dir(run_id, smoke) / "config.json"
    return read_json(path) if path.exists() else get_run(run_id, smoke)


def predict(run_id, split, beams=1, smoke=False):
    """run id -> pred/<split>_<decode>/<set>.tsv"""
    spec = run_spec(run_id, smoke)
    out_dir = run_dir(run_id, smoke)
    tag = pred_tag(split, beams)
    todo = [n for n in set_names(split) if not (out_dir / "pred" / tag / f"{n}.tsv").exists()]
    if not todo:
        print(f"{run_id} {tag} sudah ada")
        return
    if spec["kind"] == "train" and not (out_dir / "checkpoints" / "best").exists():
        print(f"{run_id}: checkpoint tidak ada di device ini, dilewati")
        return
    device = setup_device()
    tok, model = load_run_model(spec, out_dir, device)
    batch = max(1, BACKBONES[spec["backbone"]]["gen_batch"] // beams)
    zeroshot = spec["kind"] == "zeroshot"
    if zeroshot and not (out_dir / "config.json").exists():
        write_json(out_dir / "config.json", {**spec, "smoke": smoke, **device_info()})
    for name in todo:
        rows = load_set(name)[:SMOKE_EVAL_ROWS] if smoke else load_set(name)
        outs = generate(tok, model, [to_input(r["src"], spec["fmt"], zeroshot) for r in rows], batch, beams)
        write_preds(out_dir / "pred" / tag / f"{name}.tsv", rows, outs)
        print(f"{run_id} {tag} {name}: {len(rows)} baris", flush=True)
    export_run(out_dir)


def trained_runs(smoke=False):
    """run yang punya checkpoint terbaik"""
    root = SMOKE_RUNS if smoke else RUNS
    return sorted(p.name for p in root.glob("*") if (p / "checkpoints" / "best").exists())


def measure_latency(run_id, n_rows=300, smoke=False):
    """run -> latency.json (ms per baris, batch 1, conda dev)"""
    spec = run_spec(run_id, smoke)
    out_dir = run_dir(run_id, smoke)
    device = setup_device()
    tok, model = load_run_model(spec, out_dir, device)
    rows = random.Random(0).sample(load_set("conda_dev"), 20 if smoke else n_rows)
    texts = [to_input(r["src"], spec["fmt"], spec["kind"] == "zeroshot") for r in rows]
    latency(tok, model, texts[:5])  # pemanasan
    times = latency(tok, model, texts)
    result = {"rows": len(times), "p50_ms": float(np.percentile(times, 50)),
              "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times)), **device_info()}
    write_json(out_dir / "latency.json", result)
    export_run(out_dir)
    print(run_id, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", help="run id, boleh berulang")
    parser.add_argument("--all", action="store_true", help="semua run yang punya checkpoint")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_list = trained_runs(args.smoke) if args.all else args.run
    for run_id in run_list or []:
        if args.latency:
            measure_latency(run_id, smoke=args.smoke)
        else:
            predict(run_id, args.split, args.beams, args.smoke)


if __name__ == "__main__":
    main()
