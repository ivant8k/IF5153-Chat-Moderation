"""Make Notebooks: daftar run + keputusan tahap -> notebook train/predict/eval"""
import argparse
import sys
from pathlib import Path

import nbformat

NB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NB_DIR.parent / "normalizer"))

from runs import ZEROSHOT_ID, get_run, load_winners, train_runs

SETUP = '''# Environment Setup
import os
import shutil
import subprocess
import sys
from pathlib import Path

BRANCH = "feat/model/seq2seq"
REPO_URL = "https://github.com/ivant8k/IF5153-Chat-Moderation.git"
ON_COLAB = "google.colab" in sys.modules
ON_KAGGLE = Path("/kaggle/working").exists()

if ON_COLAB or ON_KAGGLE:
    if ON_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
        repo = Path("/content/repo")
        os.environ["T1_OUT"] = "/content/drive/MyDrive/Kuliah/Semester7/NLP/tugas_besar/task_1"
    else:
        repo = Path("/tmp/repo")
        os.environ["T1_OUT"] = "/kaggle/working"
        # output version sebelumnya (input) -> working
        for cfg in Path("/kaggle/input").glob("**/runs/*/config.json"):
            dst = Path("/kaggle/working/runs") / cfg.parent.name
            if not dst.exists():
                shutil.copytree(cfg.parent, dst)
    shutil.rmtree(repo, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(repo)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.22.2", "sentencepiece"],
                   check=True)
else:
    repo = Path.cwd()
    while not (repo / "pyproject.toml").exists():
        repo = repo.parent
sys.path.insert(0, str(repo / "src" / "task_1" / "normalizer"))
sys.path.insert(0, str(repo / "src" / "task_1" / "notebooks"))
print("repo:", repo, "| output:", os.environ.get("T1_OUT", "lokal"))'''

LANE_NOTE = {"local": "laptop RTX 4050 (atau CLI)", "colab": "Colab T4", "kaggle": "Kaggle T4 x2, Save Version"}


def notebook(title, note, params, body):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(f"# {title}\n\n{note}"),
                nbformat.v4.new_code_cell(params),
                nbformat.v4.new_code_cell(SETUP),
                nbformat.v4.new_code_cell(body)]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def train_notebook(run_id, smoke):
    spec = get_run(run_id, smoke)
    note = (f"Backbone `{spec['backbone']}`, data `{spec['mix']}`, upsample MLN {spec['up']}x, format `{spec['fmt']}`,"
            f" lr {spec['lr']:g}, seed {spec['seed']}.\n\nDevice: {LANE_NOTE[spec['lane']]}.")
    return notebook(run_id, note, f'RUN_ID = "{run_id}"\nSMOKE = False',
                    "# Training\nimport train\n\ntrain.train(RUN_ID, smoke=SMOKE)")


def write(path, nb, overwrite):
    # notebook yang sudah ada (bisa berisi output) tidak ditimpa
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    return path


def generate(dest=NB_DIR, smoke=False, overwrite=False):
    """keputusan tahap saat ini -> notebook yang bisa dibuat"""
    win = load_winners(smoke)
    paths = []

    # Train Notebooks
    for stage in "abcde":
        try:
            run_list = train_runs(stage, win)
        except RuntimeError:
            break
        for run_id in run_list:
            paths.append(write(dest / "train" / f"stage_{stage}" / f"{run_id}.ipynb", train_notebook(run_id, smoke), overwrite))

    # Predict Notebooks
    paths.append(write(dest / "predict" / "zeroshot.ipynb", notebook(
        "Zero-shot flan-t5-small", "Prediksi dev tanpa training. Device: laptop RTX 4050.", "SMOKE = False",
        f'# Prediction\nimport predict\n\npredict.predict("{ZEROSHOT_ID}", "dev", 1, SMOKE)'), overwrite))
    paths.append(write(dest / "predict" / "test.ipynb", notebook(
        "Prediksi test", "Dijalankan sekali setelah tahap F selesai, di tiap device untuk run miliknya.",
        "SMOKE = False",
        "# Prediction\nimport predict\nimport runs\n\n"
        "final = runs.load_winners(SMOKE)[\"f\"]\n"
        "for run_id in predict.trained_runs(SMOKE):\n"
        "    predict.predict(run_id, \"test\", 1, SMOKE)\n"
        "    # beam pemenang tahap F\n"
        "    if final[\"opts\"][\"beams\"] > 1 and run_id in runs.run_ids(final):\n"
        "        predict.predict(run_id, \"test\", final[\"opts\"][\"beams\"], SMOKE)\n"
        f"if not (ON_COLAB or ON_KAGGLE):\n    predict.predict(\"{ZEROSHOT_ID}\", \"test\", 1, SMOKE)"), overwrite))
    if "e" in win:
        paths.append(write(dest / "predict" / "stage_f.ipynb", notebook(
            "Tahap F: beam 4", "Prediksi dev beam 4 untuk pemenang tahap E, di tiap device untuk run miliknya.",
            "SMOKE = False",
            "# Prediction\nimport predict\nimport runs\n\n"
            "for run_id in runs.run_ids(runs.load_winners(SMOKE)[\"e\"]):\n"
            "    if run_id in predict.trained_runs(SMOKE):\n"
            "        predict.predict(run_id, \"dev\", 4, SMOKE)"), overwrite))

    # Eval Notebooks
    for stage in "abcdef":
        extra = "import baseline\n\nbaseline.run_baselines(SMOKE)\n" if stage == "a" else ""
        paths.append(write(dest / "eval" / f"stage_{stage}.ipynb", notebook(
            f"Eval tahap {stage.upper()}", "Membaca file prediksi di laptop. Zip dari Colab/Kaggle masuk ke "
            "`outputs/task_1/tmp/inbox/` dulu. Keputusan pemenang membuat notebook tahap berikutnya.",
            "SMOKE = False",
            "# Evaluation\nimport collect\nimport evaluate\nimport make_notebooks\n" + extra +
            "\ncollect.main()\n"
            f"evaluate.stage_report(\"{stage}\", smoke=SMOKE)\n"
            "if not SMOKE:\n    make_notebooks.generate()"), overwrite))
    paths.append(write(dest / "eval" / "final.ipynb", notebook(
        "Eval final (test)", "Membaca prediksi test semua run dan baseline.", "SMOKE = False",
        "# Evaluation\nimport collect\nimport evaluate\n\ncollect.main()\nevaluate.final_report(smoke=SMOKE)"), overwrite))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=NB_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="timpa notebook yang sudah ada")
    args = parser.parse_args()
    paths = generate(args.dest, args.smoke, args.overwrite)
    print(f"{len(paths)} notebook di {args.dest}")


if __name__ == "__main__":
    main()
