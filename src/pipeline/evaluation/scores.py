import numpy as np

from sets import GAME_SLOTS, TOXIC_INTENTS

N_BOOT = 1000
INTENTS = ["E", "I", "A", "O"]
# jenis kata: target = kata T di baris toksik (kunci sensor), sisanya tidak boleh disensor
WORD_GROUPS = ["target", "t_nontoxic", "game", "other"]


def word_group(lab, toxic_row):
    if "T" in lab:
        return "target" if toxic_row else "t_nontoxic"
    return "game" if lab & GAME_SLOTS else "other"


def row_stats(item, pred):
    """item + prediksi -> angka per baris"""
    masked = set(pred["masked_slots"])
    st = {"gate": item["gate"], "trigger": int(pred["trigger"]), "intent": item["intent"],
          "t1_valid": int(pred["t1_valid"]), "align_valid": int(pred["align_valid"]),
          "preserved": int(preserved(pred)), "labeled": 0}
    for g in WORD_GROUPS:
        st[f"n_{g}"] = st[f"masked_{g}"] = 0
    if item.get("slot_labels") is not None:
        st["labeled"] = 1
        for i, lab in enumerate(item["slot_labels"]):
            g = word_group(lab, item["intent"] in TOXIC_INTENTS)
            st[f"n_{g}"] += 1
            st[f"masked_{g}"] += int(i in masked)
    if "perturbed" in item:
        st["n_perturbed"] = len(item["perturbed"])
        st["masked_perturbed"] = len(masked & set(item["perturbed"]))
    return st


def preserved(pred):
    """karakter di luar slot tersensor sama persis dengan raw"""
    keep = set(range(len(pred["raw"])))
    for i in pred["masked_slots"]:
        _, start, end = pred["raw_slots"][i]
        keep -= set(range(start, end))
    return all(pred["raw"][i] == pred["censored"][i] for i in keep) and len(pred["raw"]) == len(pred["censored"])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def to_arrays(rows):
    """list statistik baris -> dict array"""
    keys = {k for r in rows for k in r}
    return {k: np.array([r.get(k, 0) for r in rows]) for k in keys}


def counts(a):
    """jumlah mentah: baris per intent, kata per jenis"""
    out = {}
    for k in INTENTS:
        sel = a["intent"] == k
        out[f"rows_{k}"] = int(sel.sum())
        out[f"warned_{k}"] = int(a["trigger"][sel].sum())
    for g in WORD_GROUPS:
        out[f"n_{g}"] = int(a[f"n_{g}"].sum())
        out[f"masked_{g}"] = int(a[f"masked_{g}"].sum())
    if "n_perturbed" in a:
        out["n_perturbed"] = int(a["n_perturbed"].sum())
        out["masked_perturbed"] = int(a["masked_perturbed"].sum())
    return out


def metrics(a):
    """dict array statistik baris -> dict metrik (rasio 0..1)"""
    y, t = a["gate"], a["trigger"]
    out = {k: a[k].mean() for k in ["t1_valid", "align_valid", "preserved"]}

    # Warning per Row
    gp, gr, gf = prf(int((y & t).sum()), int(((1 - y) & t).sum()), int((y & (1 - t)).sum()))
    out.update(warn_precision=gp, warn_recall=gr, warn_f1=gf)
    for k in INTENTS:
        if (a["intent"] == k).any():
            out[f"warned_rate_{k}"] = t[a["intent"] == k].mean()

    # Censor per Word
    if a["labeled"].any():
        tp = a["masked_target"].sum()
        fn = a["n_target"].sum() - tp
        fp = sum(a[f"masked_{g}"].sum() for g in WORD_GROUPS[1:])
        sp, sr, sf = prf(tp, fp, fn)
        out.update(censor_precision=sp, censor_recall=sr, censor_f1=sf)
        for g in WORD_GROUPS:
            if a[f"n_{g}"].sum():
                out[f"masked_rate_{g}"] = a[f"masked_{g}"].sum() / a[f"n_{g}"].sum()
    if "n_perturbed" in a:
        out["masked_rate_perturbed"] = a["masked_perturbed"].sum() / max(a["n_perturbed"].sum(), 1)
    return {k: float(v) for k, v in out.items()}


def take(a, idx):
    return {k: v[idx] for k, v in a.items()}


def bootstrap(a, base=None, seed=0):
    """SE bootstrap tiap metrik, atau selisih vs base berpasangan per baris + p dua sisi"""
    rng = np.random.default_rng(seed)
    n = len(a["gate"])
    samples = {}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        m = metrics(take(a, idx))
        b = metrics(take(base, idx)) if base else {}
        for k, v in m.items():
            samples.setdefault(k, []).append(v - b.get(k, 0.0))
    out = {}
    for k, v in samples.items():
        v = np.array(v)
        row = {"se": float(v.std(ddof=1)), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
        if base:
            # p dua sisi: porsi resample yang arah selisihnya berlawanan, x2
            row["identical"] = bool(np.all(v == 0))
            row["p"] = 1.0 if row["identical"] else float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))
        out[k] = row
    return out
