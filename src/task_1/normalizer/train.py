"""Train: run id -> checkpoint terbaik, prediksi dev, metrics.json"""
import argparse
import gc
import math
import random
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

from common import (export_run, load_protected, load_set, load_toxic, pred_tag, to_input, to_target, train_rows,
                    write_json, write_preds)
from config import (BACKBONES, BIG_GPU_GB, DEV_SETS, EFFECTIVE_BATCH, LOG_STEPS, MAX_EPOCHS, MAX_LEN, PATIENCE,
                    SAVE_MINUTES, SMOKE_EPOCHS, SMOKE_EVAL_ROWS, SMOKE_TRAIN_ROWS, WARMUP)
from metrics import score_set, summarize
from model import device_info, generate, load_backbone, setup_device
from runs import get_run, run_dir


def log(out_dir, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(out_dir / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Batching
def encode(tok, pairs):
    """list (input, target) -> list (input ids, label ids)"""
    src = tok([p[0] for p in pairs], truncation=True, max_length=MAX_LEN)["input_ids"]
    tgt = tok([p[1] for p in pairs], truncation=True, max_length=MAX_LEN)["input_ids"]
    return list(zip(src, tgt))


def row_len(item):
    return len(item[0]) + len(item[1])


def make_steps(encoded, seed, epoch, budget):
    """data -> list step, tiap step berisi micro-batch dengan total token <= budget"""
    order = list(range(len(encoded)))
    random.Random(seed * 1000 + epoch).shuffle(order)
    steps = []
    for start in range(0, len(order), EFFECTIVE_BATCH):
        group = sorted(order[start:start + EFFECTIVE_BATCH], key=lambda i: row_len(encoded[i]))
        micro, cur, longest = [], [], 0
        for i in group:
            longest_new = max(longest, row_len(encoded[i]))
            if cur and longest_new * (len(cur) + 1) > budget:
                micro.append(cur)
                cur, longest_new = [], row_len(encoded[i])
            cur.append(i)
            longest = longest_new
        micro.append(cur)
        steps.append(micro)
    return steps


def collate(encoded, idx, pad_id, device):
    src = [encoded[i][0] for i in idx]
    tgt = [encoded[i][1] for i in idx]
    max_src, max_tgt = max(map(len, src)), max(map(len, tgt))
    ids = torch.tensor([s + [pad_id] * (max_src - len(s)) for s in src])
    mask = torch.tensor([[1] * len(s) + [0] * (max_src - len(s)) for s in src])
    labels = torch.tensor([t + [-100] * (max_tgt - len(t)) for t in tgt])
    return ids.to(device), mask.to(device), labels.to(device)


def train_step(model, encoded, micro, pad_id, device):
    """satu step efektif -> loss rata-rata per token"""
    n_total = sum(len(encoded[i][1]) for mb in micro for i in mb)
    total = 0.0
    for mb in micro:
        ids, mask, labels = collate(encoded, mb, pad_id, device)
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        # bobot per token
        part = loss * sum(len(encoded[i][1]) for i in mb) / n_total
        part.backward()
        total += part.item()
    return total


def memory_check(model, encoded, steps, pad_id, device, base_used):
    """micro-batch terbesar -> peak VRAM (GB), peringatan jika mendekati penuh"""
    if device.type != "cuda":
        return None
    biggest = max((mb for step in steps for mb in step), key=lambda mb: max(row_len(encoded[i]) for i in mb) * len(mb))
    torch.cuda.reset_peak_memory_stats()
    ids, mask, labels = collate(encoded, biggest, pad_id, device)
    model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
    model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    if peak + base_used > 0.95 * total:
        print(f"PERINGATAN: VRAM {peak + base_used:.1f}/{total:.1f} GB, kemungkinan spill ke RAM", flush=True)
    return round(peak, 2)


# Dev Evaluation
def eval_dev(tok, model, dev, fmt, gen_batch, protected, toxic):
    """set dev -> output, metrik per set, macro-ERR"""
    outs, results = {}, {}
    for name, rows in dev.items():
        outs[name] = generate(tok, model, [to_input(r["src"], fmt) for r in rows], gen_batch)
        results[name] = summarize(score_set(rows, outs[name], fmt, protected, toxic), [r["weight"] for r in rows])
    errs = [r["err"] for r in results.values() if not math.isnan(r["err"])]
    return outs, results, float(np.mean(errs)) if errs else 0.0


# Checkpoints
def save_last(path, model, opt, sched, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "state": state,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
    tmp.replace(path)


def load_last(path, model, opt, sched, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    torch.set_rng_state(ckpt["rng"].cpu())
    if ckpt["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([r.cpu() for r in ckpt["cuda_rng"]])
    return ckpt["state"]


# Training Loop
def train(run_id, smoke=False, stop_after=None, hours=None):
    """run id -> folder run"""
    spec = get_run(run_id, smoke)
    out_dir = run_dir(run_id, smoke)
    if (out_dir / "DONE").exists():
        print(f"{run_id} sudah selesai")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # Model and Data
    device = setup_device()
    base_used = 0.0
    if device.type == "cuda":
        # sisa run sebelumnya di antrean yang sama
        gc.collect()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        base_used = (total - free) / 1e9
    tok, model = load_backbone(spec["backbone"], device)
    pairs = [(to_input(s, spec["fmt"]), to_target(t, spec["fmt"])) for s, t in train_rows(spec["mix"], spec["up"])]
    dev = {name: load_set(name) for name in DEV_SETS}
    epochs = MAX_EPOCHS
    if smoke:
        pairs = random.Random(0).sample(pairs, SMOKE_TRAIN_ROWS)
        dev = {name: rows[:SMOKE_EVAL_ROWS] for name, rows in dev.items()}
        epochs = SMOKE_EPOCHS
    encoded = encode(tok, pairs)
    protected, toxic = load_protected(), load_toxic()
    info = BACKBONES[spec["backbone"]]
    big_gpu = device.type == "cuda" and torch.cuda.get_device_properties(0).total_memory / 1e9 >= BIG_GPU_GB
    budget = info["budget"] * (2 if big_gpu else 1)

    # Optimizer
    steps_per_epoch = math.ceil(len(encoded) / EFFECTIVE_BATCH)
    opt = Adafactor(model.parameters(), lr=spec["lr"], relative_step=False, scale_parameter=False, warmup_init=False)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps_per_epoch * epochs), steps_per_epoch * epochs)

    # Resume
    last_path = out_dir / "checkpoints" / "last.pt"
    if last_path.exists():
        state = load_last(last_path, model, opt, sched, device)
        log(out_dir, f"resume epoch {state['epoch']} step {state['step']}")
    else:
        peak = memory_check(model, encoded, make_steps(encoded, spec["seed"], 0, budget), tok.pad_token_id,
                            device, base_used)
        torch.manual_seed(spec["seed"])
        state = {"epoch": 0, "step": 0, "global_step": 0, "best": None, "best_epoch": None, "bad": 0,
                 "best_results": None, "history": [], "train_sec": 0.0, "peak_vram_gb": peak}
        write_json(out_dir / "config.json", {**spec, **info, "budget": budget, "smoke": smoke, "epochs": epochs,
                                              "n_train": len(encoded), "steps_per_epoch": steps_per_epoch,
                                              **device_info()})
        log(out_dir, f"mulai {run_id} | train {len(encoded)} baris | {steps_per_epoch} step/epoch | budget {budget}")

    # Epochs
    while state["epoch"] < epochs and state["bad"] < PATIENCE:
        model.train()
        steps = make_steps(encoded, spec["seed"], state["epoch"], budget)
        last_save, loss_sum, loss_n = time.time(), 0.0, 0
        t0 = time.time()
        for i in range(state["step"], len(steps)):
            loss_sum += train_step(model, encoded, steps[i], tok.pad_token_id, device)
            loss_n += 1
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            state["step"], state["global_step"] = i + 1, state["global_step"] + 1
            if state["global_step"] % LOG_STEPS == 0:
                log(out_dir, f"epoch {state['epoch']} step {i + 1}/{len(steps)} loss {loss_sum / loss_n:.4f}")
            stop = (stop_after and state["global_step"] >= stop_after) or (hours and time.time() - started > hours * 3600)
            if stop or time.time() - last_save > SAVE_MINUTES * 60:
                state["train_sec"] += time.time() - t0
                t0 = time.time()
                save_last(last_path, model, opt, sched, state)
                last_save = time.time()
            if stop:
                log(out_dir, f"berhenti di epoch {state['epoch']} step {i + 1}, lanjutkan dengan run yang sama")
                return out_dir
        state["train_sec"] += time.time() - t0

        # Epoch Evaluation
        outs, results, macro = eval_dev(tok, model, dev, spec["fmt"], info["gen_batch"], protected, toxic)
        state["history"].append({"epoch": state["epoch"], "loss": loss_sum / max(loss_n, 1), "macro_err": macro,
                                 **{f"err_{k}": v["err"] for k, v in results.items()}})
        if state["best"] is None or macro > state["best"]:
            state.update(best=macro, best_epoch=state["epoch"], bad=0, best_results=results)
            model.save_pretrained(out_dir / "checkpoints" / "best")
            tok.save_pretrained(out_dir / "checkpoints" / "best")
            for name, rows in dev.items():
                write_preds(out_dir / "pred" / pred_tag("dev", 1) / f"{name}.tsv", rows, outs[name])
        else:
            state["bad"] += 1
        log(out_dir, f"epoch {state['epoch']} selesai | loss {loss_sum / max(loss_n, 1):.4f} | macro-ERR {macro:.4f}"
                     f" | terbaik {state['best']:.4f} (epoch {state['best_epoch']})")
        state["epoch"], state["step"] = state["epoch"] + 1, 0
        save_last(last_path, model, opt, sched, state)

    # Finalization
    write_json(out_dir / "metrics.json", {"run": spec, "best_epoch": state["best_epoch"], "macro_err": state["best"],
                                          "dev": state["best_results"], "history": state["history"],
                                          "train_hours": round(state["train_sec"] / 3600, 3),
                                          "peak_vram_gb": state["peak_vram_gb"], **device_info()})
    last_path.unlink(missing_ok=True)
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    export_run(out_dir)
    log(out_dir, f"selesai {run_id} | macro-ERR {state['best']:.4f} | epoch terbaik {state['best_epoch']}")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="run id, boleh berulang")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after", type=int, help="berhenti setelah N step (uji resume)")
    parser.add_argument("--hours", type=float, help="batas waktu antrean")
    args = parser.parse_args()
    started = time.time()
    for run_id in args.run:
        left = None if args.hours is None else args.hours - (time.time() - started) / 3600
        if left is not None and left <= 0:
            break
        out_dir = train(run_id, args.smoke, args.stop_after, left)
        if not (out_dir / "DONE").exists():
            break


if __name__ == "__main__":
    main()
