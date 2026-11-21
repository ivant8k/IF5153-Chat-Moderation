"""Baseline E0 Task 2: TF-IDF (kata + karakter) + Logistic Regression

Data sama dengan E2 (Jigsaw + CONDA). Gate: semua data. Kategori: hanya baris dengan mask=1 (sama seperti masked loss).
python src/task_2/baseline.py --run b0_tfidf
"""
import argparse
import pickle
import time

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import build_train, write_json
from config import CATS, LABELS, RUNS, SEED


class TfidfPredictor:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.vec_w, self.vec_c, self.models = pickle.load(f)

    def probs(self, texts):
        X = hstack([self.vec_w.transform(texts), self.vec_c.transform(texts)]).tocsr()
        out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        for k, clf in enumerate(self.models):
            out[:, k] = clf.predict_proba(X)[:, 1]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="b0_tfidf")
    ap.add_argument("--sources", default="jigsaw,conda")
    ap.add_argument("--conda-text", default="norm,raw")
    ap.add_argument("--jigsaw-neg-frac", type=float, default=0.35)
    ap.add_argument("--c", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    t0 = time.time()
    texts, Y, M = build_train(a.sources.split(","), a.conda_text.split(","), a.jigsaw_neg_frac, SEED, a.limit)
    print(f"train {len(texts)}", flush=True)
    vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200000, sublinear_tf=True)
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=300000, sublinear_tf=True)
    X = hstack([vec_w.fit_transform(texts), vec_c.fit_transform(texts)]).tocsr()

    models = []
    for k, name in enumerate(LABELS):
        valid = M[:, k] > 0
        clf = LogisticRegression(C=a.c, solver="liblinear", max_iter=200)
        clf.fit(X[valid], Y[valid, k].astype(int))
        models.append(clf)
        print(f"  {name}: {valid.sum()} baris, positif {int(Y[valid, k].sum())}", flush=True)

    out_dir = RUNS / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump((vec_w, vec_c, models), f)
    write_json(out_dir / "config.json", {**vars(a), "kind": "tfidf", "n_train": len(texts), "seconds": time.time() - t0})
    (out_dir / "DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
