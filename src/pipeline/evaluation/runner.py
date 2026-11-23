"""Runner evaluasi: set uji x varian -> prediksi, metrik + SE bootstrap, waktu proses"""
import json
import time

import numpy as np
import torch

from loader import task_scope
from scores import bootstrap, counts, metrics, row_stats, to_arrays
from sets import SETS, conda_bench, tokenize
from settings import DELIM, MASK, OUT, T3_DIR

# varian: normalisasi, gate T2, mode ambang
VARIANTS = {
    "full": ("t1", True, "balanced"),
    "strict": ("t1", True, "strict"),
    "no_t1": ("none", True, "balanced"),
    "oracle_t1": ("oracle", True, "balanced"),
    "no_gate": ("t1", False, "balanced"),
    "lexicon": None,
}
RESULTS = OUT / "results"
LATENCY_ROWS = 300


def semeval_lexicon():
    """kamus kata toksik dari SemEval train (baseline Task 3)"""
    with task_scope(T3_DIR):
        import dataset
        import model
        return model.build_toxic_lexicon(dataset.load_semeval_split("train"))


def join_slots(lines):
    """list baris chat -> slot raw dengan offset di teks gabungan"""
    out, offset = [], 0
    for line in lines:
        out += [(t, s + offset, e + offset) for t, s, e in tokenize(line)]
        offset += len(line) + 1
    return out


def run_lexicon(items, lexicon):
    """baseline tanpa model: slot raw ada di kamus -> sensor, ada sensor -> warning"""
    preds = []
    for it in items:
        raw, raw_slots = " ".join(it["lines"]), join_slots(it["lines"])
        masked = [i for i, (t, _, _) in enumerate(raw_slots) if t.lower() in lexicon]
        chars = list(raw)
        for i in masked:
            _, s, e = raw_slots[i]
            chars[s:e] = MASK * (e - s)
        preds.append({"raw": raw, "src": DELIM.join(t.lower() for t, _, _ in raw_slots), "norm": "",
                      "t1_valid": True, "p_toxic": float(bool(masked)), "trigger": bool(masked), "warnings": [],
                      "tags": [], "masked_slots": masked, "align_valid": True, "censored": "".join(chars)})
    return preds


def save_preds(path, items, preds):
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = ["raw", "src", "norm", "t1_valid", "p_toxic", "trigger", "warnings", "tags", "masked_slots",
            "align_valid", "censored"]
    with open(path, "w", encoding="utf-8") as f:
        for it, p in zip(items, preds):
            row = {"id": it["id"], "intent": it["intent"], **{k: p[k] for k in keep}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_predictions(mod, set_names=tuple(SETS), variants=tuple(VARIANTS)):
    """set uji x varian -> outputs/pipeline/pred/<set>/<varian>.jsonl, detik per langkah"""
    lexicon = semeval_lexicon() if "lexicon" in variants else None
    seconds = {}
    for name in set_names:
        items = SETS[name]()
        rows = [it["lines"] for it in items]
        start = time.perf_counter()
        t1_norm, t1_valid = mod.normalize(rows)
        seconds[f"{name}|t1"] = time.perf_counter() - start
        for var in variants:
            start = time.perf_counter()
            if var == "lexicon":
                preds = run_lexicon(items, lexicon)
            else:
                norm, gate, mode = VARIANTS[var]
                if norm == "oracle":
                    norm, valid = [it["norm"] for it in items], None
                elif norm == "t1":
                    norm, valid = t1_norm, t1_valid
                else:
                    valid = None
                mod.set_mode(mode)
                preds, _ = mod.run(rows, norm, valid, gate)
            seconds[f"{name}|{var}"] = time.perf_counter() - start
            save_preds(OUT / "pred" / name / f"{var}.jsonl", items, preds)
        mod.set_mode("balanced")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "seconds.json").write_text(json.dumps(seconds, indent=1) + "\n", encoding="utf-8")
    return seconds


def run_scores(set_names=tuple(SETS), variants=tuple(VARIANTS)):
    """prediksi tersimpan -> metrics.tsv (metrik + jumlah mentah), bootstrap.json (SE, uji beda vs full)"""
    table, boot = [], {}
    for name in set_names:
        items = SETS[name]()
        arrays = {}
        for var in variants:
            path = OUT / "pred" / name / f"{var}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                preds = {r["id"]: r for r in map(json.loads, f)}
            stats = []
            for it in items:
                p = preds[it["id"]]
                p["raw_slots"] = join_slots(it["lines"])
                stats.append(row_stats(it, p))
            a = arrays[var] = to_arrays(stats)
            table += [{"set": name, "variant": var, "metric": k, "value": v}
                      for k, v in {**metrics(a), **counts(a)}.items()]
        for var, a in arrays.items():
            boot[f"{name}|{var}"] = {"se": bootstrap(a)}
            if var != "full" and "full" in arrays:
                boot[f"{name}|{var}"]["delta_vs_full"] = bootstrap(a, arrays["full"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "metrics.tsv", "w", encoding="utf-8") as f:
        f.write("set\tvariant\tmetric\tvalue\n")
        for r in table:
            f.write(f"{r['set']}\t{r['variant']}\t{r['metric']}\t{r['value']:.6f}\n")
    (RESULTS / "bootstrap.json").write_text(json.dumps(boot, indent=1) + "\n", encoding="utf-8")


def run_latency(mod, n_rows=LATENCY_ROWS, seed=2026):
    """ms per pesan (satu baris per panggilan) per tahap -> latency_<device>.json"""
    items = conda_bench()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=min(n_rows, len(items)), replace=False)
    if mod.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    mod.moderate("warm up")
    per = {"t1": [], "t2": [], "t3": [], "total": []}
    for i in idx:
        # tanpa gate: waktu T3 batas atas
        _, times = mod.run([items[i]["lines"]], gate=False)
        for k, v in times.items():
            per[k].append(v * 1000)
        per["total"].append(sum(times.values()) * 1000)
    out = {"device": str(mod.device), "n": len(idx)}
    if mod.device.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
        out["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    for k, v in per.items():
        v = np.array(v)
        out[k] = {"mean": round(float(v.mean()), 1), "p50": round(float(np.percentile(v, 50)), 1),
                  "p95": round(float(np.percentile(v, 95)), 1)}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"latency_{mod.device.type}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out
