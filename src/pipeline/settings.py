from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
T1_PREPARE_DIR = SRC / "task_1" / "prepare"
T1_DIR = SRC / "task_1" / "normalizer"
T2_DIR = SRC / "task_2"
T3_DIR = SRC / "task_3"
OUT = ROOT / "outputs" / "pipeline"

# Final Models
T1_RUN = "a_byt5-small_s2026"
T2_RUN = "e2_mix"
T2_MODE = "balanced"
T3_CKPT = ROOT / "outputs" / "task_3" / "runs" / "baseline" / "bert_crf.pt"
T3_BACKBONE = ROOT / "models" / "pretrained" / "bert-base-uncased"

# Text Format
DELIM = " | "
SEPA = "[SEPA]"
MASK = "*"
