"""Smoke Notebooks: notebook (SMOKE = True) -> eksekusi headless berurutan per tahap"""
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR.parent / "normalizer"))

from config import OUT
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
    for sub in ("runs", "results", "notebooks"):
        shutil.rmtree(OUT / "tmp" / "smoke" / sub, ignore_errors=True)
    for stage in "abcd":
        generate(TMP_NB, smoke=True)
        run_folder(TMP_NB / "train" / f"stage_{stage}")
        if stage == "a":
            execute(TMP_NB / "predict" / "zeroshot.ipynb")
        execute(TMP_NB / "eval" / f"stage_{stage}.ipynb")
    generate(TMP_NB, smoke=True)
    for name in ("predict/stage_e", "eval/stage_e", "predict/test", "eval/final"):
        execute(TMP_NB / f"{name}.ipynb")
    # checkpoint smoke tidak dipakai lagi
    for ckpt in (OUT / "tmp" / "smoke" / "runs").glob("*/checkpoints"):
        shutil.rmtree(ckpt)
    print("smoke notebook selesai")


if __name__ == "__main__":
    main()
