"""Calibrate Task 2: ambang gate (balanced/strict) di dev CONDA, ambang kategori di dev Jigsaw, ECE"""
import numpy as np
from sklearn.metrics import precision_recall_curve

from config import CATS, MIN_POS_FOR_THRESHOLD, STRICT_MIN_PRECISION


def pr_thresholds(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    return prec[:-1], rec[:-1], thr


def best_f1_threshold(y, p, default=0.5):
    if y.sum() < MIN_POS_FOR_THRESHOLD:
        return default
    prec, rec, thr = pr_thresholds(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def strict_threshold(y, p, min_precision=STRICT_MIN_PRECISION):
    """ambang terendah dengan precision >= min_precision (recall terbesar); jika tak tercapai, maks F0.5"""
    prec, rec, thr = pr_thresholds(y, p)
    ok = np.where(prec >= min_precision)[0]
    if len(ok):
        return float(thr[ok[0]])
    f05 = 1.25 * prec * rec / np.clip(0.25 * prec + rec, 1e-9, None)
    return float(thr[int(np.argmax(f05))])


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def calibrate(conda_dev_y, conda_dev_p, jigsaw_dev_y=None, jigsaw_dev_p=None):
    """-> thresholds.json: {gate: {balanced, strict}, cats: {label: thr}}"""
    balanced = best_f1_threshold(conda_dev_y, conda_dev_p)
    out = {"gate": {"balanced": balanced,
                    "strict": max(balanced, strict_threshold(conda_dev_y, conda_dev_p))}, "cats": {}}
    for k, c in enumerate(CATS):
        if jigsaw_dev_y is None:
            out["cats"][c] = 0.5
        else:
            out["cats"][c] = best_f1_threshold(jigsaw_dev_y[:, k], jigsaw_dev_p[:, 1 + k])
    return out
