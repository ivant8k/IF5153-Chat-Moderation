"""Collect: zip run dari Colab/Kaggle (tmp/inbox) -> outputs/task_1/runs"""
import shutil
import zipfile

from config import INBOX, ROOT

RUNS_LOCAL = ROOT / "outputs" / "task_1" / "runs"


def main():
    INBOX.mkdir(parents=True, exist_ok=True)
    zips = sorted(INBOX.glob("*.zip"))
    if not zips:
        print(f"inbox kosong: {INBOX}")
        return
    for path in zips:
        run_id = path.stem
        if (RUNS_LOCAL / run_id / "DONE").exists():
            print(f"{run_id} sudah ada, dilewati")
        else:
            with zipfile.ZipFile(path) as z:
                z.extractall(RUNS_LOCAL)
            print(f"{run_id} masuk ke {RUNS_LOCAL / run_id}")
        done = INBOX / "done"
        done.mkdir(exist_ok=True)
        shutil.move(path, done / path.name)


if __name__ == "__main__":
    main()
