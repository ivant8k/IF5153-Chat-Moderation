"""Collect: zip run dari Colab/Kaggle (tmp/inbox) -> outputs/task_1/runs"""
import shutil
import zipfile

from config import INBOX, ROOT

RUNS_LOCAL = ROOT / "outputs" / "task_1" / "runs"


def merge(path):
    """zip run -> file yang belum ada di folder run lokal (file lama tidak ditimpa)"""
    added = 0
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if member.endswith("/") or (RUNS_LOCAL / member).exists():
                continue
            z.extract(member, RUNS_LOCAL)
            added += 1
    return added


def main():
    INBOX.mkdir(parents=True, exist_ok=True)
    zips = sorted(INBOX.glob("*.zip"))
    if not zips:
        print(f"inbox kosong: {INBOX}")
        return
    for path in zips:
        print(f"{path.stem}: {merge(path)} file baru di {RUNS_LOCAL / path.stem}")
        done = INBOX / "done"
        done.mkdir(exist_ok=True)
        shutil.move(path, done / path.name)


if __name__ == "__main__":
    main()
