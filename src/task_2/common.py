"""Common Task 2: teks, TSV, load set, builder data latih"""
import csv
import json
import re

import numpy as np
import pandas as pd

from config import CATS, DELIM, LABELS, OUT

WS = re.compile(r"\s+")


def clean_text(s):
    """Teks input Task 2: buang delimiter Task 1, lowercase (output Task 1 selalu lowercase), rapikan spasi"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).replace(DELIM, " ").replace("\x01", " ")
    return WS.sub(" ", s.lower()).strip()


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def write_tsv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, quotechar="\x01")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_set(name):
    """data/processed/task_2/<name>.tsv; kolom label jadi int"""
    df = read_tsv(OUT / f"{name}.tsv")
    for c in ["gate"] + CATS:
        if c in df.columns:
            df[c] = df[c].astype(int)
    return df


def build_train(sources, conda_text, neg_frac, seed, limit=None):
    """sumber -> (texts, Y, M). Y: target (n, 7), M: mask loss (n, 7).

    Jigsaw: semua label di-supervisi.
    CONDA bukan toksik (A/O): gate=0 dan semua kategori=0 di-supervisi.
    CONDA toksik (E/I): hanya gate=1; kategori di-mask (tidak ada label kategori).
    """
    rng = np.random.default_rng(seed)
    texts, ys, ms = [], [], []

    if "jigsaw" in sources:
        j = load_set("jigsaw_train")
        keep = (j["gate"].values == 1) | (rng.random(len(j)) < neg_frac)
        j = j[keep]
        if limit:
            j = j.sample(min(limit, len(j)), random_state=seed)
        texts += j["text"].tolist()
        ys.append(j[LABELS].values.astype(np.float32))
        ms.append(np.ones((len(j), len(LABELS)), dtype=np.float32))

    if "conda" in sources:
        c = load_set("conda_train")
        if limit:
            c = c.sample(min(limit, len(c)), random_state=seed)
        for variant in conda_text:
            texts += c[variant].tolist()
            y = np.zeros((len(c), len(LABELS)), dtype=np.float32)
            y[:, 0] = c["gate"].values
            m = np.ones((len(c), len(LABELS)), dtype=np.float32)
            m[c["gate"].values == 1, 1:] = 0.0
            ys.append(y)
            ms.append(m)

    Y, M = np.concatenate(ys), np.concatenate(ms)
    ok = np.array([len(t) > 0 for t in texts])
    return [t for t, k in zip(texts, ok) if k], Y[ok], M[ok]
