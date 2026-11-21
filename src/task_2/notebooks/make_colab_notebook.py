"""Make Colab notebook Task 2 -> src/task_2/notebooks/colab_runner.ipynb"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "colab_runner.ipynb"

SETUP = '''# Setup: Colab + Drive. Jigsaw (train.csv, test.csv, test_labels.csv) taruh di Drive
import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/ivant8k/IF5153-Chat-Moderation.git"
BRANCH = "main"
DRIVE_JIGSAW = "/content/drive/MyDrive/tubes_nlp/jigsaw"      # folder berisi 3 csv Jigsaw
DRIVE_OUT = "/content/drive/MyDrive/tubes_nlp/task_2_runs"    # checkpoint + hasil (awet)

from google.colab import drive
drive.mount("/content/drive")
if not Path("/content/repo").exists():
    subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, "/content/repo"], check=True)
os.chdir("/content/repo")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "scikit-learn", "pandas"], check=True)
Path("data/raw").mkdir(parents=True, exist_ok=True)
if not Path("data/raw/conda").exists():
    subprocess.run("git clone --depth 1 https://github.com/usydnlp/CONDA.git data/raw/conda && rm -rf data/raw/conda/.git", shell=True, check=True)
if not Path("data/raw/jigsaw").exists():
    os.symlink(DRIVE_JIGSAW, "data/raw/jigsaw")
Path(DRIVE_OUT).mkdir(parents=True, exist_ok=True)
Path("outputs/task_2").mkdir(parents=True, exist_ok=True)
if not Path("outputs/task_2/runs").exists():
    os.symlink(DRIVE_OUT, "outputs/task_2/runs")
!nvidia-smi | head -15'''

CELLS = [
    ("md", "# Task 2: Toxicity Classification (Colab runner)\nJalankan sel berurutan. Run yang sudah `DONE` dilewati, jadi aman diulang setelah runtime putus."),
    ("code", SETUP),
    ("md", "## 1. Data\nJigsaw + CONDA (split Task 1) + NoisyHate -> `data/processed/task_2`"),
    ("code", "!python src/task_2/prepare.py"),
    ("md", "## 2. Baseline\nE0 TF-IDF + LR dan zero-shot `unitary/toxic-bert`"),
    ("code", "!python src/task_2/baseline.py --run b0_tfidf\n!python src/task_2/evaluate.py --run b0_tfidf\n!python src/task_2/evaluate.py --run z_toxicbert --kind zeroshot"),
    ("md", "## 3. Arm DistilRoBERTa\nE1 Jigsaw saja, E2 campuran (kandidat utama), E3 bertahap (Jigsaw lalu CONDA)"),
    ("code", "!python src/task_2/train.py --run e1_jigsaw --sources jigsaw\n!python src/task_2/evaluate.py --run e1_jigsaw"),
    ("code", "!python src/task_2/train.py --run e2_mix --sources jigsaw,conda\n!python src/task_2/evaluate.py --run e2_mix"),
    ("code", "!python src/task_2/train.py --run e3_seq --sources conda --init-from e1_jigsaw --lr 1e-5 --select all\n!python src/task_2/evaluate.py --run e3_seq"),
    ("md", "## 4. Ringkasan"),
    ("code", "!python src/task_2/summary.py"),
    ("md", "## 5. Demo kontrak output (ganti run sesuai pemenang)"),
    ("code", 'import sys\nsys.path.insert(0, "src/task_2")\nfrom predict import ToxicityClassifier\nimport json\nclf = ToxicityClassifier("e2_mix", mode="balanced")\nfor r in clf.classify(["you are idiot", "gg wp", "report mid he is feeding", "u | r | trash | noob"]):\n    print(json.dumps(r, indent=1))'),
]


def cell(kind, src):
    base = {"metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "md":
        return {"cell_type": "markdown", **base}
    return {"cell_type": "code", "execution_count": None, "outputs": [], **base}


nb = {"cells": [cell(k, s) for k, s in CELLS], "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print("tulis", OUT)
