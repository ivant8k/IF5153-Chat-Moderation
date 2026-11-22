"""Evaluate Task 2: probabilitas semua set -> kalibrasi ambang -> metrik (trigger, kategori, ketahanan NoisyHate)

python src/task_2/evaluate.py --run e2_mix                       # checkpoint torch
python src/task_2/evaluate.py --run b0_tfidf --kind tfidf        # baseline TF-IDF
python src/task_2/evaluate.py --run z_toxicbert --kind zeroshot  # model siap pakai
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score

from calibrate import calibrate, ece
from common import load_set, read_json, write_json
from config import CATS, RUNS, ZEROSHOT_ID


def get_predictor(run, kind, hf_id):
    if kind == "torch":
        from model import TorchPredictor
        return TorchPredictor(RUNS / run / "checkpoints" / "best")
    if kind == "tfidf":
        from baseline import TfidfPredictor
        return TfidfPredictor(RUNS / run / "model.pkl")
    from model import ZeroShotPredictor
    return ZeroShotPredictor(hf_id)


def eval_sets(limit):
    """kunci -> (df, kolom teks)"""
    sets = {}
    for name in ["conda_dev", "conda_bench"]:
        d = load_set(name)
        for v in ["norm", "raw", "t1pred"]:
            if v in d.columns:
                sets[f"{name}:{v}"] = (d, v)
    gold = load_set("conda_gold")
    for v in ["gold", "t1pred"]:
        sets[f"conda_gold:{v}"] = (gold, v)
    sets["jigsaw_dev"] = (load_set("jigsaw_dev"), "text")
    sets["jigsaw_test"] = (load_set("jigsaw_test"), "text")
    nh = load_set("noisyhate")
    for v in ["perturbed", "t1pred", "clean"]:
        sets[f"noisyhate:{v}"] = (nh, v)
    if limit:
        sets = {k: (d.sample(min(limit, len(d)), random_state=0), c) for k, (d, c) in sets.items()}
    return sets


def gate_metrics(y, p, thr, intent=None):
    pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    neg = y == 0
    out = {"precision": pr, "recall": rc, "f1": f1, "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
           "fpr": float(pred[neg].mean()) if neg.any() else 0.0, "flag_rate": float(pred.mean())}
    if intent is not None:
        for k in ["E", "I"]:
            m = intent == k
            out[f"recall_{k}"] = float(pred[m].mean()) if m.any() else np.nan
        for k in ["A", "O"]:
            m = intent == k
            out[f"fpr_{k}"] = float(pred[m].mean()) if m.any() else np.nan
    return out


def evaluate(run, kind, hf_id, limit=None, reuse=False):
    run_dir = RUNS / run
    sets = eval_sets(limit)
    predictor = None

    probs = {}
    for key, (df, col) in sets.items():
        path = run_dir / "probs" / f"{key.replace(':', '__')}.npy"
        if reuse and path.exists() and len(np.load(path)) == len(df):
            probs[key] = np.load(path)
            continue
        predictor = predictor or get_predictor(run, kind, hf_id)
        probs[key] = predictor.probs(df[col].tolist())
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, probs[key])
        print(f"{run} {key}: {len(df)} baris", flush=True)

    jd = sets["jigsaw_dev"][0]
    cd = sets["conda_dev:norm"][0]
    th = calibrate(cd["gate"].values, probs["conda_dev:norm"][:, 0], jd[CATS].values, probs["jigsaw_dev"])
    write_json(run_dir / "thresholds.json", th)

    rows = []

    def add(key, metric, value):
        rows.append({"set": key, "metric": metric, "value": float(value)})

    for key, (df, col) in sets.items():
        p = probs[key]
        if key.startswith("conda"):
            y = df["gate"].values
            add(key, "gate_ap", average_precision_score(y, p[:, 0]))
            add(key, "gate_roc_auc", roc_auc_score(y, p[:, 0]))
            add(key, "gate_ece", ece(y, p[:, 0]))
            for mode in ["balanced", "strict"]:
                for m, v in gate_metrics(y, p[:, 0], th["gate"][mode], df["intent"].values).items():
                    add(key, f"{mode}_{m}", v)
        elif key.startswith("jigsaw"):
            y = df[CATS].values
            aucs, aps, f1s = [], [], []
            for k, c in enumerate(CATS):
                if y[:, k].sum() == 0:
                    continue
                auc, ap = roc_auc_score(y[:, k], p[:, 1 + k]), average_precision_score(y[:, k], p[:, 1 + k])
                f1 = f1_score(y[:, k], (p[:, 1 + k] >= th["cats"][c]).astype(int), zero_division=0)
                add(key, f"{c}_auc", auc), add(key, f"{c}_ap", ap), add(key, f"{c}_f1", f1)
                aucs.append(auc), aps.append(ap), f1s.append(f1)
            add(key, "mean_auc", np.mean(aucs)), add(key, "macro_ap", np.mean(aps)), add(key, "macro_f1", np.mean(f1s))
            add(key, "gate_ap", average_precision_score(df["gate"], p[:, 0]))
            add(key, "gate_roc_auc", roc_auc_score(df["gate"], p[:, 0]))
        else:   # noisyhate: semua toksik -> flag_rate = recall
            for mode in ["balanced", "strict"]:
                add(key, f"{mode}_recall", (p[:, 0] >= th["gate"][mode]).mean())
            add(key, "mean_p_gate", p[:, 0].mean())

    m = pd.DataFrame(rows)
    m.to_csv(run_dir / "metrics.tsv", sep="\t", index=False)
    write_json(run_dir / "metrics.json", {f"{r.set}|{r.metric}": r.value for r in m.itertuples()})
    show = m[m["metric"].isin(["gate_ap", "gate_roc_auc", "balanced_f1", "balanced_macro_f1", "balanced_fpr_A",
                               "balanced_fpr_O", "balanced_recall_E", "balanced_recall_I", "strict_precision",
                               "strict_recall", "mean_auc", "macro_ap", "macro_f1", "balanced_recall", "strict_recall"])]
    print(show.pivot_table(index="set", columns="metric", values="value").round(3).T.to_string())
    print("thresholds:", th)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--kind", choices=["torch", "tfidf", "zeroshot"], default=None)
    ap.add_argument("--hf-id", default=ZEROSHOT_ID)
    ap.add_argument("--limit", type=int, default=None, help="subsampel tiap set (uji cepat)")
    ap.add_argument("--reuse", action="store_true", help="pakai probs tersimpan, hitung ulang kalibrasi + metrik saja")
    a = ap.parse_args()
    cfg = RUNS / a.run / "config.json"
    kind = a.kind or (read_json(cfg)["kind"] if cfg.exists() else "torch")
    evaluate(a.run, kind, a.hf_id, a.limit, a.reuse)
