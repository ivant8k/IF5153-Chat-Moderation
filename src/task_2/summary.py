"""Summary Task 2: bandingkan semua run -> outputs/task_2/summary.tsv"""
import pandas as pd

from config import RUNS

KEYS = [
    ("conda_bench:norm", "gate_ap"), ("conda_bench:norm", "balanced_f1"), ("conda_bench:norm", "balanced_macro_f1"),
    ("conda_bench:norm", "balanced_fpr_A"), ("conda_bench:norm", "balanced_fpr_O"),
    ("conda_bench:norm", "balanced_recall_E"), ("conda_bench:norm", "balanced_recall_I"),
    ("conda_bench:norm", "strict_precision"), ("conda_bench:norm", "strict_recall"),
    ("conda_bench:raw", "balanced_f1"), ("conda_bench:t1pred", "balanced_f1"),
    ("conda_gold:gold", "gate_ap"), ("conda_gold:gold", "balanced_f1"),
    ("conda_gold:t1pred", "balanced_f1"),
    ("jigsaw_test", "mean_auc"), ("jigsaw_test", "macro_ap"), ("jigsaw_test", "macro_f1"),
    ("noisyhate:perturbed", "balanced_recall"), ("noisyhate:t1pred", "balanced_recall"),
    ("noisyhate:clean", "balanced_recall"),
]

if __name__ == "__main__":
    table = {}
    for p in sorted(RUNS.glob("*/metrics.tsv")):
        m = pd.read_csv(p, sep="\t").set_index(["set", "metric"])["value"]
        table[p.parent.name] = {f"{s}|{k}": m.get((s, k), float("nan")) for s, k in KEYS}
    df = pd.DataFrame(table)
    df.round(4).to_csv(RUNS.parent / "summary.tsv", sep="\t")
    pd.set_option("display.width", 250)
    print(df.round(3).to_string())
