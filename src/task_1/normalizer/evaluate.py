"""Evaluate: file prediksi -> tabel tahap, uji bootstrap, keputusan pemenang"""
import argparse
import math
import time

import numpy as np

from common import load_protected, load_set, load_toxic, pred_tag, read_json, read_preds, set_names, write_json, write_tsv
from config import MIN_VALIDITY, N_BOOT, NONINF_MARGIN, SMOKE_EVAL_ROWS
from metrics import boot_delta, ci_and_p, err_arrays, holm, score_set, summarize
from runs import BASELINES, ZEROSHOT_ID, load_winners, need, results_dir, run_dir, run_ids, stage_configs

DEFAULT_OPTS = {"beams": 1, "whitelist": False, "repair": False}
EXTRA_METRICS = ["acc", "precision", "recall", "f1", "validity", "overnorm", "overnorm_protected", "toxic_acc",
                 "obf_acc"]


class Context:
    """cache set data + resource"""

    def __init__(self, smoke):
        self.smoke = smoke
        self.protected, self.toxic = load_protected(), load_toxic()
        self.sets = {}

    def rows(self, name):
        if name not in self.sets:
            rows = load_set(name)
            self.sets[name] = rows[:SMOKE_EVAL_ROWS] if self.smoke else rows
        return self.sets[name]


def run_scores(run_id, split, opts, ctx):
    """run -> {set: skor baris}, None jika prediksi belum lengkap"""
    out_dir = run_dir(run_id, ctx.smoke)
    if not (out_dir / "config.json").exists():
        return None
    fmt = read_json(out_dir / "config.json")["fmt"]
    result = {}
    for name in set_names(split):
        preds = read_preds(out_dir / "pred" / pred_tag(split, opts["beams"]) / f"{name}.tsv")
        if preds is None:
            return None
        rows = ctx.rows(name)
        outs = [preds.get(r["id"], "") for r in rows]
        result[name] = score_set(rows, outs, fmt, ctx.protected, ctx.toxic, opts["whitelist"], opts["repair"])
    return result


def macro(values):
    vals = [v for v in values if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def summarize_config(name, runs, split, opts, ctx):
    """config + run per seed -> ringkasan + array untuk bootstrap"""
    per_seed = {rid: s for rid in runs if (s := run_scores(rid, split, opts, ctx)) is not None}
    if not per_seed:
        return None
    sets = set_names(split)
    metrics = {rid: {n: summarize(s[n], [r["weight"] for r in ctx.rows(n)], [r.get("op") for r in ctx.rows(n)])
                     for n in sets} for rid, s in per_seed.items()}
    seed_macro = {rid: macro([m[n]["err"] for n in sets]) for rid, m in metrics.items()}
    arrays = []
    for n in sets:
        weights = [r["weight"] for r in ctx.rows(n)]
        per = [err_arrays(s[n], weights) for s in per_seed.values()]
        # correct dirata-rata antar seed sebelum bootstrap
        arrays.append([np.mean([p[0] for p in per], axis=0), per[0][1], per[0][2]])
    row = {"config": name, "seeds": len(per_seed), "macro_err": np.mean(list(seed_macro.values())),
           "macro_err_std": np.std(list(seed_macro.values())),
           "validity": np.mean([m[n]["validity"] for m in metrics.values() for n in sets])}
    for n in sets:
        row[f"err_{n}"] = np.mean([m[n]["err"] for m in metrics.values()])
    long = [{"config": name, "set": n, "metric": k, "mean": np.mean([m[n][k] for m in metrics.values()]),
             "std": np.std([m[n][k] for m in metrics.values()])}
            for n in sets for k in metrics[next(iter(metrics))][n] if k != "rows"]
    return {"row": row, "long": long, "arrays": arrays, "seed_macro": seed_macro}


def seed_of(run_id):
    return run_id.rsplit("_s", 1)[-1]


def compare(best, others, margin):
    """pemenang skor vs config lain -> baris uji (CI, p Holm, konsistensi seed)"""
    pairs = []
    for c in others:
        samples = boot_delta(c["arrays"], best["arrays"], N_BOOT)
        low, high, p = ci_and_p(samples)
        best_seeds = {seed_of(r): v for r, v in best["seed_macro"].items()}
        common = [(best_seeds[seed_of(r)], v) for r, v in c["seed_macro"].items() if seed_of(r) in best_seeds]
        ok = low > -margin if margin is not None else high >= 0
        pairs.append({"config": c["row"]["config"], "vs": best["row"]["config"], "delta": float(samples.mean()),
                      "ci_low": low, "ci_high": high, "p": p, "best_better_seeds": sum(b > v for b, v in common),
                      "common_seeds": len(common), "acceptable": ok})
    for pair, p_adj in zip(pairs, holm([p["p"] for p in pairs])):
        pair["p_holm"] = p_adj
    return pairs


def fmt_row(row):
    return {k: f"{v:.4f}" if isinstance(v, (float, np.floating)) else v for k, v in row.items()}


def write_report(folder, summaries, refs, pairs, decision):
    rows = [fmt_row(s["row"]) for s in summaries + refs]
    write_tsv(folder / "table.tsv", rows, list(rows[0]))
    long = [fmt_row(x) for s in summaries + refs for x in s["long"]]
    write_tsv(folder / "metrics.tsv", long, ["config", "set", "metric", "mean", "std"])
    if pairs:
        write_tsv(folder / "pairs.tsv", [fmt_row(p) for p in pairs], list(pairs[0]))
    if decision:
        write_json(folder / "decision.json", decision)
    for r in rows:
        print("  ".join(f"{k}={v}" for k, v in r.items()))
    for p in pairs:
        print(f"{p['config']} vs {p['vs']}: delta {p['delta']:+.4f} CI [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]"
              f" p_holm {p['p_holm']:.3f} seed {p['best_better_seeds']}/{p['common_seeds']}")


# Stage Configurations
def stage_e_configs(win):
    """pemenang tahap D -> varian inferensi"""
    prev = need(win, "d")
    out = []
    for beams in (1, 4):
        for wl in (0, 1):
            for rp in (0, 1):
                name = f"{'greedy' if beams == 1 else f'beam{beams}'}-wl{wl}-rp{rp}"
                out.append({"name": name, "prefix": prev["prefix"], "params": prev["params"],
                            "cost": (beams > 1) + wl + rp,
                            "opts": {"beams": beams, "whitelist": bool(wl), "repair": bool(rp)}})
    return out


def stage_report(stage, smoke=False):
    """tahap -> results/stage_x/{table, metrics, pairs}.tsv + decision.json"""
    ctx = Context(smoke)
    win = load_winners(smoke)
    configs = stage_e_configs(win) if stage == "e" else stage_configs(stage, win)
    summaries = []
    for c in configs:
        s = summarize_config(c["name"], run_ids(c), "dev", c.get("opts", DEFAULT_OPTS), ctx)
        if s is None:
            print(f"lewati {c['name']}: prediksi belum ada")
            continue
        s["config"] = c
        summaries.append(s)
    refs = []
    if stage == "a":
        for rid in [ZEROSHOT_ID] + BASELINES:
            s = summarize_config(rid, [rid], "dev", DEFAULT_OPTS, ctx)
            if s is not None:
                refs.append(s)
    if not summaries:
        raise RuntimeError(f"tahap {stage}: belum ada run dengan prediksi dev")

    # Winner Selection
    valid = [s for s in summaries if s["row"]["validity"] >= MIN_VALIDITY] or summaries
    best = max(valid, key=lambda s: s["row"]["macro_err"])
    margin = NONINF_MARGIN if stage == "c" else None
    pairs = compare(best, [s for s in valid if s is not best], margin)
    ok = {p["config"] for p in pairs if p["acceptable"]} | {best["row"]["config"]}
    winner = min((s for s in valid if s["row"]["config"] in ok), key=lambda s: s["config"]["cost"])
    c = winner["config"]
    decision = {"stage": stage, "winner": c["name"], "prefix": c["prefix"], "params": c["params"],
                "best_score": best["row"]["config"], "macro_err": float(winner["row"]["macro_err"]),
                "seeds": winner["row"]["seeds"], "date": time.strftime("%Y-%m-%d %H:%M")}
    if stage == "e":
        decision["opts"] = c["opts"]
    write_report(results_dir(smoke) / f"stage_{stage}", summaries, refs, pairs, decision)
    print(f"pemenang tahap {stage}: {c['name']} ({c['prefix']})")
    return decision


def final_report(smoke=False):
    """semua config + baseline -> results/final (test set)"""
    ctx = Context(smoke)
    win = load_winners(smoke)
    final = need(win, "e")
    groups, seen = [], set()
    for stage in "abcd":
        for c in stage_configs(stage, win):
            if c["prefix"] not in seen:
                seen.add(c["prefix"])
                groups.append((f"{stage}:{c['name']}", run_ids(c), DEFAULT_OPTS))
    groups.append(("final", run_ids(final), final["opts"]))
    groups += [(rid, [rid], DEFAULT_OPTS) for rid in [ZEROSHOT_ID] + BASELINES]

    summaries = [s for name, runs, opts in groups if (s := summarize_config(name, runs, "test", opts, ctx))]
    by_name = {s["row"]["config"]: s for s in summaries}
    if "final" not in by_name:
        raise RuntimeError("prediksi test pemenang belum ada")

    # Final vs Baselines
    pairs = []
    for ref in ["base_mfr-all", "base_rule", ZEROSHOT_ID]:
        if ref in by_name:
            for p in compare(by_name[ref], [by_name["final"]], None):
                p["config"], p["vs"] = "final", ref
                pairs.append(p)
    write_report(results_dir(smoke) / "final", summaries, [], pairs, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["a", "b", "c", "d", "e", "final"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.stage == "final":
        final_report(args.smoke)
    else:
        stage_report(args.stage, args.smoke)


if __name__ == "__main__":
    main()
