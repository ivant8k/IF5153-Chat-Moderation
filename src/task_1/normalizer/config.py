"""Config normalizer: path, backbone, hyperparameter"""
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "task_1"
ANN = ROOT / "data" / "annotations" / "task_1"
VOCAB_FILE = ROOT / "data" / "raw" / "multilexnorm" / "wortschatz" / "en" / "eng_news_2025_1M-words.txt"
WHITELIST_FILE = DATA / "resources" / "whitelist.txt"
TOXIC_FILE = DATA / "resources" / "toxic.txt"
PRETRAINED = ROOT / "models" / "pretrained"
OUT = Path(os.environ.get("T1_OUT", ROOT / "outputs" / "task_1"))
RUNS = OUT / "runs"
SMOKE_RUNS = OUT / "tmp" / "smoke" / "runs"
SMOKE_RESULTS = OUT / "tmp" / "smoke" / "results"
EXPORT = OUT / "export"
RESULTS = ROOT / "outputs" / "task_1" / "results"
INBOX = ROOT / "outputs" / "task_1" / "tmp" / "inbox"

# Backbones
BACKBONES = {
    "t5-small": {"hf": "google-t5/t5-small", "rev": "df1b051c49625cf57a3d0d8d3863ed4d13564fe4",
                 "budget": 4096, "gen_batch": 64, "cost": 1},
    "flan-t5-small": {"hf": "google/flan-t5-small", "rev": "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
                      "budget": 4096, "gen_batch": 64, "cost": 2},
    "byt5-small": {"hf": "google/byt5-small", "rev": "68377bdc18a2ffec8a0533fef03b1c513a4dd49d",
                   "budget": 1024, "gen_batch": 32, "cost": 3},
}
ZEROSHOT_BACKBONE = "flan-t5-small"
PROXIES = {"byt5-small": "flan-t5-small"}    # pemenang A -> backbone tahap B-D

# Data Sets
TRAIN_FILES = {"mln": "mln/train.tsv", "conda": "conda/train.tsv", "aug": "augmented/train.tsv"}
MIXES = {"a": ["mln"], "b": ["mln", "conda"], "c": ["mln", "conda", "aug"]}
DEV_SETS = {"mln_dev": "mln/dev.tsv", "conda_dev": "conda/dev.tsv", "aug_dev": "augmented/conda_dev.tsv"}
TEST_SETS = {"mln_test": "mln/test.tsv", "gold_conda": "gold", "conda_bench": "conda/bench.tsv",
             "aug_bench": "augmented/conda_bench.tsv", "noisyhate": "noisyhate/test.tsv"}

# Text Format
DELIM = " | "
PREFIX = "normalize: "
ZEROSHOT_PROMPT = "Rewrite this chat message in standard English spelling: "

# Training Configurations
SEEDS = (2026, 13, 42)
LANES = {2026: "local", 13: "colab", 42: "kaggle"}
BASE = {"mix": "c", "up": 1, "fmt": "delim", "lr": 1e-3}
EFFECTIVE_BATCH = 32
MAX_EPOCHS = 10
PATIENCE = 2
WARMUP = 0.05
MAX_LEN = 256
SAVE_MINUTES = 20
LOG_STEPS = 200
BIG_GPU_GB = 12        # budget x2 di GPU >= 12 GB

# Selection Configurations
MIN_VALIDITY = 0.99
N_BOOT = 10000
NONINF_MARGIN = 0.01

# Smoke Configurations
SMOKE_TRAIN_ROWS = 96
SMOKE_EVAL_ROWS = 16
SMOKE_EPOCHS = 2
