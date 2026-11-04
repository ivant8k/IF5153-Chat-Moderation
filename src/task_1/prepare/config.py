"""Config Task 1: path + konstanta preprocessing"""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
ANN = ROOT / "data" / "annotations" / "task_1"
OUT = ROOT / "data" / "processed" / "task_1"

# Raw Data
MLN_DIR = RAW / "multilexnorm"
VOCAB_FILE = MLN_DIR / "wortschatz" / "en" / "eng_news_2025_1M-words.txt"
CONDA_DIR = RAW / "conda"
NOISYHATE_FILE = RAW / "noisyhate" / "toxic detection model test set with perturbations.csv"

# Annotations
ANN_TYPE = ANN / "type_labels"
ANN_SLOT = ANN / "slot_labels"
ANN_LINE = ANN / "line_labels"
ANN_RES = ANN / "resources"

# Processed Data
OUT_MLN = OUT / "mln"
OUT_CONDA = OUT / "conda"
OUT_NOISYHATE = OUT / "noisyhate"
OUT_AUG = OUT / "augmented"
OUT_LABELING = OUT / "labeling"
OUT_RES = OUT / "resources"
OUT_REPORT = OUT / "reports"

# Split Configurations
SEED = 2026
SPLIT_RATIO = (("train", 0.8), ("dev", 0.1), ("bench", 0.1))

# Label Source Configurations
MIN_FREQ_TYPE = 5      # tipe sering, dilabel per tipe
MIN_FREQ_VOCAB = 20    # korpus berita masih ada typo

# Gold Configurations
GOLD_QUOTA = (("toxic", 90), ("changed", 90), ("game", 60), ("other", 60))

# Augmentation Configurations
AUG_RATIO = 0.3        # porsi baris train
TOXIC_WEIGHT = 3
AUG_OPS = (("repeat", 0.25), ("leet", 0.25), ("symbol", 0.15), ("drop", 0.10),
           ("accent", 0.10), ("spaced", 0.10), ("leet_repeat", 0.05))

# Format
DELIM = " | "
SEPA = "[SEPA]"
PIPE_SUB = "/"         # pengganti '|' di teks
