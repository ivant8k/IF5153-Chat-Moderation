"""Config Task 2: path + konstanta klasifikasi toksisitas"""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
JIGSAW_DIR = RAW / "jigsaw"
CONDA_DIR = RAW / "conda"
T1_PROC = ROOT / "data" / "processed" / "task_1"
T1_RUNS = ROOT / "outputs" / "task_1" / "runs"
T1_GOLD = ROOT / "data" / "annotations" / "task_1" / "line_labels" / "gold_conda_v2.tsv"
OUT = ROOT / "data" / "processed" / "task_2"
RUNS = ROOT / "outputs" / "task_2" / "runs"

# Task 1 prediction run yang dipakai sebagai input end-to-end (ByT5-small final, seed 42)
T1_PRED_RUN = "a_byt5-small_s42"

# Labels: index 0 = gate (toksik apa pun), 1..6 = kategori Jigsaw
CATS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
LABELS = ["gate"] + CATS

# Data
SEED = 2026
DELIM = " | "           # delimiter Task 1
SEPA = "[SEPA]"         # pemisah chat dalam satu baris CONDA
JIGSAW_DEV_FRAC = 0.05
TOXIC_INTENTS = {"E", "I"}   # CONDA: Explicit, Implicit -> toksik; A (Action), O (Other) -> bukan

# Model
BACKBONE = "distilroberta-base"
MAX_LEN = 128
ZEROSHOT_ID = "unitary/toxic-bert"   # baseline siap pakai, label sama dengan Jigsaw

# Calibration
STRICT_MIN_PRECISION = 0.95
MIN_POS_FOR_THRESHOLD = 20

# Structured warning
WARN_NAMES = {
    "toxic": "toxic language",
    "severe_toxic": "severely toxic language",
    "obscene": "obscene language",
    "threat": "threats",
    "insult": "insults",
    "identity_hate": "identity-based hate",
}
