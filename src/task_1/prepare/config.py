"""Config Task 1: path + konstanta preprocessing"""
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
ANN = ROOT / "data" / "annotations" / "task_1"
OUT = ROOT / "data" / "processed" / "task_1"

MLN_DIR = RAW / "multilexnorm"
VOCAB_FILE = MLN_DIR / "wortschatz" / "en" / "eng_news_2025_1M-words.txt"
CONDA_DIR = RAW / "conda"
NOISYHATE_FILE = RAW / "noisyhate" / "toxic detection model test set with perturbations.csv"

# Split Configurations
SEED = 2026
SPLIT_RATIO = (("train", 0.8), ("dev", 0.1), ("bench", 0.1))

# Funnel Configurations
MIN_FREQ_TYPE = 5      # lapis 6: tipe sering, dilabel per tipe
MIN_FREQ_VOCAB = 20    # lapis 5: korpus berita masih mengandung typo frekuensi kecil

# Format
DELIM = " | "
SEPA = "[SEPA]"
PIPE_SUB = "/"         # '|' di teks diganti agar tidak bentrok dengan delimiter
