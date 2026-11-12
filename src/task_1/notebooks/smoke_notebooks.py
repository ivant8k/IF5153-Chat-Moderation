"""Smoke Notebooks: notebook (SMOKE = True) -> eksekusi headless berurutan per tahap"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR.parent / "normalizer"))

from common import write_json
from config import OUT, SMOKE_RESULTS
from runs import a_config
from make_notebooks import generate

TMP_NB = OUT / "tmp" / "smoke" / "notebooks"


def execute(path):
    nb = nbformat.read(path, as_version=4)
    nb.cells[1].source = nb.cells[1].source.replace("SMOKE = False", "SMOKE = True")
    start = time.time()
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    print(f"ok {time.time() - start:5.1f}s {path.relative_to(TMP_NB)}", flush=True)


def run_folder(folder):
    for path in sorted(folder.glob("*.ipynb")):
        execute(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner-a", help="paksa pemenang tahap A, untuk menguji jalur proxy")
    args = parser.parse_args()
    for sub in ("runs", "results", "notebooks"):
        shutil.rmtree(OUT / "tmp" / "smoke" / sub, ignore_errors=True)
    for stage in "abcde":
        generate(TMP_NB, smoke=True)
        run_folder(TMP_NB / "train" / f"stage_{stage}")
        if stage == "a":
            execute(TMP_NB / "predict" / "zeroshot.ipynb")
        execute(TMP_NB / "eval" / f"stage_{stage}.ipynb")
        if stage == "a" and args.winner_a:
            write_json(SMOKE_RESULTS / "stage_a" / "decision.json", {"stage": "a", **a_config(args.winner_a)})
    generate(TMP_NB, smoke=True)
    for name in ("predict/stage_f", "eval/stage_f", "predict/test", "eval/final"):
        execute(TMP_NB / f"{name}.ipynb")
    # checkpoint smoke tidak dipakai lagi
    for ckpt in (OUT / "tmp" / "smoke" / "runs").glob("*/checkpoints"):
        shutil.rmtree(ckpt)
    print("smoke notebook selesai")


if __name__ == "__main__":
    main()
