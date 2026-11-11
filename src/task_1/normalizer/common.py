"""Common: TSV, set data, format teks, resource"""
import csv
import json
import os
import zipfile
from collections import Counter

from config import (ANN, DATA, DELIM, DEV_SETS, EXPORT, LEXICON_FILE, MIXES, PREFIX, TEST_SETS, TRAIN_FILES,
                    VOCAB_FILE, WHITELIST_FILE, ZEROSHOT_PROMPT)


# File IO
def read_tsv(path):
    """TSV -> list dict"""
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split("\t")
    return [dict(zip(fields, line.split("\t"))) for line in lines[1:] if line]


def write_tsv(path, rows, fields):
    """list dict -> TSV"""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = ["\t".join(fields)]
    for row in rows:
        values = [str(row.get(k, "")) for k in fields]
        assert not any(c in v for v in values for c in "\t\n\r"), row
        out.append("\t".join(values))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def split_slots(line):
    return line.split(DELIM)


# Prediction Files
def pred_tag(split, beams):
    return f"{split}_greedy" if beams == 1 else f"{split}_beam{beams}"


def write_preds(path, rows, outs):
    write_tsv(path, [{"id": r["id"], "out": o} for r, o in zip(rows, outs)], ["id", "out"])


def read_preds(path):
    """file prediksi -> dict id: output"""
    return {r["id"]: r["out"] for r in read_tsv(path)} if path.exists() else None


def export_run(out_dir):
    """folder run tanpa checkpoint -> export/<run_id>.zip (cloud saja)"""
    if not os.environ.get("T1_OUT"):
        return
    EXPORT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(EXPORT / f"{out_dir.name}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in out_dir.rglob("*"):
            if f.is_file() and "checkpoints" not in f.parts:
                z.write(f, f.relative_to(out_dir.parent))


# Evaluation Sets
def load_gold():
    """gold_conda -> rows dengan bobot strata"""
    rows = read_tsv(ANN / "line_labels" / "gold_conda.tsv")
    population = read_json(ANN / "line_labels" / "gold_strata.json")
    sampled = Counter(r["stratum"] for r in rows)
    return [{"id": r["id"], "src": r["src"], "tgt": r["gold"], "stratum": r["stratum"],
             "weight": population[r["stratum"]] / sampled[r["stratum"]]} for r in rows]


def load_set(name):
    """nama set -> list row (id, src, tgt, weight, obf_slot, op)"""
    path = {**DEV_SETS, **TEST_SETS}[name]
    if path == "gold":
        return load_gold()
    rows = []
    for r in read_tsv(DATA / path):
        row = {"id": r["id"], "src": r["src"], "tgt": r["tgt"], "weight": 1.0}
        # slot yang disamarkan
        if "slot" in r:
            row["obf_slot"], row["op"] = int(r["slot"]), r["op"]
        if r.get("changed"):
            row["obf_slot"], row["op"] = int(r["changed"]), "real"
        rows.append(row)
    return rows


def set_names(split):
    return list(DEV_SETS) if split == "dev" else list(TEST_SETS)


# Training Data
def train_rows(mix, up):
    """mix + upsample MLN -> list (src, tgt)"""
    rows = []
    for part in MIXES[mix]:
        pairs = [(r["src"], r["tgt"]) for r in read_tsv(DATA / TRAIN_FILES[part])]
        rows += pairs * (up if part == "mln" else 1)
    return rows


# Text Format
def to_input(src, fmt, zeroshot=False):
    """src slot -> teks input model"""
    if zeroshot:
        return ZEROSHOT_PROMPT + " ".join(split_slots(src))
    if fmt == "plain":
        return PREFIX + " ".join(split_slots(src))
    return PREFIX + src


def to_target(tgt, fmt):
    return " ".join(split_slots(tgt)) if fmt == "plain" else tgt


# Resources
def load_protected():
    return {w.strip() for w in WHITELIST_FILE.read_text(encoding="utf-8").splitlines() if w.strip()}


def load_toxic():
    with open(LEXICON_FILE, encoding="utf-8", newline="") as f:
        return {row["T"].lower() for row in csv.DictReader(f) if row["T"]}


def load_vocab(min_freq=20):
    """Wortschatz -> set kata baku"""
    freq = Counter()
    for line in VOCAB_FILE.read_text(encoding="utf-8").splitlines():
        _, word, n = line.split("\t")
        freq[word.lower()] += int(n)
    return {w for w, n in freq.items() if n >= min_freq}
