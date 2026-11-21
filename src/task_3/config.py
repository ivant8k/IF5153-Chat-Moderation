"""Task 3 configuration: paths, labels, tokenization, and seeds."""
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "semeval2021"
DATA_DIR = RAW_DIR / "data"
EVAL_DIR = RAW_DIR / "evaluation"
DATA_FILES = {
    "train": DATA_DIR / "tsd_train.csv",
    "trial": DATA_DIR / "tsd_trial.csv",
    "test": DATA_DIR / "tsd_test.csv",
}

OUT_DIR = Path(os.environ.get("T3_OUT", ROOT / "outputs" / "task_3"))
RUNS_DIR = OUT_DIR / "runs"
RESULTS_DIR = OUT_DIR / "results"

DEFAULT_BACKBONE = "bert-base-uncased"
MAX_LEN = 256

LABEL2ID = {"O": 0, "B-TOXIC": 1, "I-TOXIC": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_TAGS = len(LABEL2ID)

SEEDS = (2026, 13, 42)
DEFAULT_SEED = SEEDS[0]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Lock random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
