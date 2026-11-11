"""Metrics: slot prediksi -> skor per baris, ringkasan, bootstrap"""
import numpy as np

from align import postprocess
from common import split_slots

COUNT_KEYS = ["n", "correct", "lai", "pred_changed", "gold_changed", "tp", "keep_n", "keep_changed",
              "prot_n", "prot_changed", "tox_n", "tox_ok", "obf_n", "obf_ok", "valid"]


def score_row(row, pred, valid, protected, toxic):
    """row + slot prediksi -> dict hitungan"""
    src, tgt = split_slots(row["src"]), split_slots(row["tgt"])
    s = dict.fromkeys(COUNT_KEYS, 0)
    s["n"], s["valid"] = len(src), int(valid)
    for x, t, p in zip(src, tgt, pred):
        s["correct"] += p == t
        s["lai"] += x == t
        s["pred_changed"] += p != x
        s["gold_changed"] += t != x
        s["tp"] += p == t and t != x
        if t == x:
            s["keep_n"] += 1
            s["keep_changed"] += p != x
        if x in protected and t == x:
            s["prot_n"] += 1
            s["prot_changed"] += p != x
        if any(w in toxic for w in t.split()):
            s["tox_n"] += 1
            s["tox_ok"] += p == t
    if "obf_slot" in row:
        k = row["obf_slot"]
        s["obf_n"], s["obf_ok"] = 1, int(pred[k] == tgt[k])
    return s


def score_set(rows, outs, fmt, protected, toxic, whitelist=False, repair=False):
    """rows + output model -> list skor baris"""
    scores = []
    for row, out in zip(rows, outs):
        pred, valid = postprocess(row["src"], out, fmt, protected if whitelist else None, repair)
        scores.append(score_row(row, pred, valid, protected, toxic))
    return scores


def ratio(a, b):
    return a / b if b else float("nan")


def err(acc, lai_acc):
    return ratio(acc - lai_acc, 1 - lai_acc)


def summarize(scores, weights, ops=None):
    """list skor baris -> dict metrik"""
    w = np.array(weights)
    tot = {k: float((np.array([s[k] for s in scores]) * w).sum()) for k in COUNT_KEYS}
    acc, lai_acc = ratio(tot["correct"], tot["n"]), ratio(tot["lai"], tot["n"])
    prec, rec = ratio(tot["tp"], tot["pred_changed"]), ratio(tot["tp"], tot["gold_changed"])
    out = {"rows": len(scores), "acc": acc, "lai_acc": lai_acc, "err": err(acc, lai_acc),
           "precision": prec, "recall": rec, "f1": ratio(2 * prec * rec, prec + rec),
           "validity": ratio(tot["valid"], w.sum()),
           "overnorm": ratio(tot["keep_changed"], tot["keep_n"]),
           "overnorm_protected": ratio(tot["prot_changed"], tot["prot_n"]),
           "toxic_acc": ratio(tot["tox_ok"], tot["tox_n"]),
           "obf_acc": ratio(tot["obf_ok"], tot["obf_n"])}
    # akurasi slot tersamar per operator
    if ops:
        for op in sorted(set(o for o in ops if o)):
            idx = [i for i, o in enumerate(ops) if o == op]
            out[f"obf_acc_{op}"] = ratio(sum(scores[i]["obf_ok"] for i in idx), len(idx))
    return out


# Bootstrap
def err_arrays(scores, weights):
    """skor baris -> array correct, lai, n (berbobot)"""
    w = np.array(weights)
    return [np.array([s[k] for s in scores], dtype=float) * w for k in ("correct", "lai", "n")]


def boot_err(arrays, idx):
    correct, lai, n = (a[idx].sum(axis=-1) for a in arrays)
    # resample tanpa slot berubah -> ERR 0
    return np.divide(correct - lai, n - lai, out=np.zeros_like(correct), where=(n - lai) > 0)


def boot_delta(sets_a, sets_b, n_boot, seed=0):
    """per set (arrays A, arrays B) -> sampel delta macro-ERR (A - B)"""
    rng = np.random.default_rng(seed)
    total = np.zeros(n_boot)
    for arr_a, arr_b in zip(sets_a, sets_b):
        n_rows = len(arr_a[0])
        for start in range(0, n_boot, 500):
            size = min(500, n_boot - start)
            idx = rng.integers(0, n_rows, size=(size, n_rows))
            total[start:start + size] += boot_err(arr_a, idx) - boot_err(arr_b, idx)
    return total / len(sets_a)


def ci_and_p(samples):
    low, high = np.percentile(samples, [2.5, 97.5])
    p = min(1.0, 2 * min((samples <= 0).mean(), (samples >= 0).mean()))
    return float(low), float(high), float(p)


def holm(pvalues):
    """list p -> p terkoreksi Holm-Bonferroni"""
    order = np.argsort(pvalues)
    out, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        out[i] = running
    return out
