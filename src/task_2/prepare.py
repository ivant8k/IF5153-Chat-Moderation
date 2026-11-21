"""Prepare Task 2: Jigsaw + CONDA (split Task 1) + NoisyHate -> data/processed/task_2

Jigsaw : data/raw/jigsaw/{train,test,test_labels}.csv
CONDA  : label intent dari data/raw/conda, teks + split dari Task 1 (match-level, seed 2026)
         csv test CONDA tidak berlabel -> dibuang
"""
import re

import numpy as np
import pandas as pd

from common import clean_text, read_tsv, write_tsv
from config import (CATS, CONDA_DIR, JIGSAW_DEV_FRAC, JIGSAW_DIR, OUT, SEED, T1_GOLD, T1_PRED_RUN, T1_PROC,
                    T1_RUNS, TOXIC_INTENTS)

ID_RE = re.compile(r"^conda-(\d+)-(\d+)$")


def load_t1_predictions(filename):
    path = T1_RUNS / T1_PRED_RUN / "pred" / "test_greedy" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Prediksi Task 1 tidak ditemukan: {path}")
    predictions = read_tsv(path)
    if not {"id", "out"}.issubset(predictions.columns):
        raise ValueError(f"{path} harus memiliki kolom id dan out")
    if predictions["id"].duplicated().any():
        raise ValueError(f"ID prediksi Task 1 duplikat di {path}")
    return dict(zip(predictions["id"], predictions["out"]))


def prepare_jigsaw():
    tr = pd.read_csv(JIGSAW_DIR / "train.csv")
    te = pd.read_csv(JIGSAW_DIR / "test.csv").merge(pd.read_csv(JIGSAW_DIR / "test_labels.csv"), on="id")
    te = te[te["toxic"] != -1]   # baris tanpa label

    def fmt(df):
        out = pd.DataFrame({"id": df["id"].values, "text": [clean_text(t)[:1000] for t in df["comment_text"]]})
        for c in CATS:
            out[c] = df[c].astype(int).values
        out["gate"] = (df[CATS].sum(axis=1) > 0).astype(int).values
        return out[out["text"] != ""]

    tr, te = fmt(tr), fmt(te)
    dev = tr.sample(frac=JIGSAW_DEV_FRAC, random_state=SEED)
    tr = tr.drop(dev.index)
    for name, df in [("jigsaw_train", tr), ("jigsaw_dev", dev), ("jigsaw_test", te)]:
        write_tsv(df, OUT / f"{name}.tsv")
        print(f"{name}: {len(df)} | gate+ {df['gate'].mean():.3f}")


def prepare_conda():
    raw = pd.concat([pd.read_csv(CONDA_DIR / "data" / f"CONDA_{s}.csv") for s in ["train", "valid"]])
    raw = raw.dropna(subset=["utterance"]).drop_duplicates("Id").set_index("Id")

    pred = load_t1_predictions("conda_bench.tsv")

    for split in ["train", "dev", "bench"]:
        lines = read_tsv(T1_PROC / "conda" / f"{split}.tsv")
        rows = {}
        for r in lines.itertuples():
            m = ID_RE.match(r.id)
            rid, k = int(m.group(1)), int(m.group(2))
            rows.setdefault(rid, []).append((k, r.id, r.raw, r.tgt))
        out = []
        for rid, parts in rows.items():
            if rid not in raw.index:      # baris dari csv test (tanpa label)
                continue
            parts.sort()
            missing = [part_id for _, part_id, _, _ in parts if part_id not in pred]
            if split == "bench" and missing:
                raise ValueError(f"Prediksi Task 1 hilang untuk ID CONDA bench: {missing[:5]}")
            intent = raw.loc[rid, "intentClass"]
            row = {"id": f"conda-{rid}", "match_id": raw.loc[rid, "matchId"], "intent": intent,
                   "gate": int(intent in TOXIC_INTENTS),
                   "raw": clean_text(" ".join(p[2] for p in parts), ),
                   "norm": clean_text(" ".join(p[3] for p in parts))}
            if split == "bench":
                row["t1pred"] = clean_text(" ".join(pred[p[1]] for p in parts))
            out.append(row)
        df = pd.DataFrame(out)
        df = df[(df["raw"] != "") & (df["norm"] != "")].sort_values("id")
        write_tsv(df, OUT / f"conda_{split}.tsv")
        print(f"conda_{split}: {len(df)} baris | gate+ {df['gate'].mean():.3f} | intent {df['intent'].value_counts().to_dict()}")

    prepare_conda_gold(raw)


def prepare_conda_gold(raw):
    gold = read_tsv(T1_GOLD)
    required = {"id", "raw", "silver", "gold", "checked"}
    if not required.issubset(gold.columns):
        raise ValueError(f"{T1_GOLD} harus memiliki kolom {', '.join(sorted(required))}")
    gold = gold[gold["checked"] == "1"].copy()
    if gold["id"].duplicated().any():
        raise ValueError(f"ID duplikat di {T1_GOLD}")
    predictions = load_t1_predictions("gold_conda.tsv")
    rows = []
    for row in gold.itertuples(index=False):
        match = ID_RE.match(row.id)
        if match is None:
            raise ValueError(f"ID CONDA gold tidak valid: {row.id}")
        conda_id = int(match.group(1))
        if conda_id not in raw.index:
            raise ValueError(f"ID CONDA gold tidak memiliki label intent: {row.id}")
        if row.id not in predictions:
            raise ValueError(f"Prediksi Task 1 hilang untuk ID CONDA gold: {row.id}")
        intent = raw.loc[conda_id, "intentClass"]
        rows.append({
            "id": row.id,
            "intent": intent,
            "gate": int(intent in TOXIC_INTENTS),
            "raw": clean_text(row.raw),
            "silver": clean_text(row.silver),
            "gold": clean_text(row.gold),
            "t1pred": clean_text(predictions[row.id]),
        })
    if not rows:
        raise ValueError(f"Tidak ada baris gold yang sudah diperiksa di {T1_GOLD}")
    df = pd.DataFrame(rows)
    write_tsv(df, OUT / "conda_gold.tsv")
    print(f"conda_gold: {len(df)} baris | gate+ {df['gate'].mean():.3f}")


def prepare_noisyhate():
    """perturbed (raw), clean (kunci), t1pred. Semua kalimat diasumsikan toksik"""
    nh = read_tsv(T1_PROC / "noisyhate" / "test.tsv")
    pred = load_t1_predictions("noisyhate.tsv")
    missing = sorted(set(nh["id"]) - pred.keys())
    if missing:
        raise ValueError(f"Prediksi Task 1 hilang untuk ID NoisyHate: {missing[:5]}")
    df = pd.DataFrame({"id": nh["id"], "perturbed": [clean_text(t) for t in nh["raw"]],
                       "clean": [clean_text(t) for t in nh["tgt"]],
                       "t1pred": [clean_text(pred[i]) for i in nh["id"]]})
    df["gate"] = 1
    write_tsv(df, OUT / "noisyhate.tsv")
    print(f"noisyhate: {len(df)} baris")


if __name__ == "__main__":
    np.random.seed(SEED)
    prepare_jigsaw()
    prepare_conda()
    prepare_noisyhate()
