"""Prepare: raw MLN, CONDA, NoisyHate -> TSV slot + split CONDA per match"""
import csv
import random

from common import clean_slot, mln_slots, read_norm, to_src, tokenize, write_tsv
from config import (CONDA_DIR, DELIM, MLN_DIR, NOISYHATE_FILE, OUT_CONDA, OUT_MLN, OUT_NOISYHATE, SEED,
                    SEPA, SPLIT_RATIO)

MLN_FIELDS = ["id", "split", "source", "raw", "src", "tgt"]
CONDA_FIELDS = ["id", "split", "match_id", "raw", "src"]
NOISYHATE_FIELDS = MLN_FIELDS + ["changed"]


def prepare_mln():
    for split in ["train", "dev", "test"]:
        rows = []
        for i, pairs in enumerate(read_norm(MLN_DIR / "en" / f"{split}.norm")):
            src, tgt = mln_slots(pairs)
            rows.append({
                "id": f"mln-{split}-{i}",
                "split": split,
                "source": "mln",
                "raw": " ".join(raw for raw, _ in pairs),
                "src": DELIM.join(src),
                "tgt": DELIM.join(tgt),
            })
        write_tsv(OUT_MLN / f"{split}.tsv", rows, MLN_FIELDS)
        print(f"mln {split}: {len(rows)} kalimat")


def split_matches(match_ids):
    """match id -> split"""
    matches = sorted(match_ids, key=int)
    random.Random(SEED).shuffle(matches)
    assign, start = {}, 0
    for i, (split, ratio) in enumerate(SPLIT_RATIO):
        end = len(matches) if i == len(SPLIT_RATIO) - 1 else start + round(ratio * len(matches))
        for m in matches[start:end]:
            assign[m] = split
        start = end
    return assign


def prepare_conda():
    rows = []
    for csv_name in ["train", "valid", "test"]:
        with open(CONDA_DIR / "data" / f"CONDA_{csv_name}.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.append((csv_name, r))
    assign = split_matches({r["matchId"] for _, r in rows})

    lines, dropped = [], 0
    for csv_name, r in rows:
        split = assign[r["matchId"]]
        # bench hanya dari csv berlabel
        if split == "bench" and csv_name == "test":
            dropped += 1
            continue
        parts = [p.strip() for p in (r["utterance"] or "").split(SEPA)]
        for k, raw in enumerate(parts):
            if raw:
                lines.append({"id": f"conda-{r['Id']}-{k}", "split": split,
                              "match_id": r["matchId"], "raw": raw, "src": to_src(raw)})
    lines.sort(key=lambda x: (x["split"], int(x["id"].split("-")[1]), int(x["id"].split("-")[2])))
    write_tsv(OUT_CONDA / "lines.tsv", lines, CONDA_FIELDS)

    n_match = {s: sum(1 for v in assign.values() if v == s) for s, _ in SPLIT_RATIO}
    n_line = {s: sum(1 for x in lines if x["split"] == s) for s, _ in SPLIT_RATIO}
    print(f"conda match: {n_match}")
    print(f"conda baris chat: {n_line} | utterance test-csv di bench dibuang: {dropped}")


def prepare_noisyhate():
    """perturbed -> clean, sejajar per slot"""
    rows, dropped = [], 0
    with open(NOISYHATE_FILE, encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            src = [clean_slot(t) for t, _, _ in tokenize(r["perturbed_version"])]
            tgt = [clean_slot(t) for t, _, _ in tokenize(r["clean_version"])]
            if len(src) != len(tgt):
                dropped += 1
                continue
            changed = [str(k) for k, (s, t) in enumerate(zip(src, tgt)) if s != t]
            rows.append({"id": f"noisyhate-{i}", "split": "test", "source": "noisyhate",
                         "raw": r["perturbed_version"], "src": DELIM.join(src), "tgt": DELIM.join(tgt),
                         "changed": ",".join(changed)})
    write_tsv(OUT_NOISYHATE / "test.tsv", rows, NOISYHATE_FIELDS)
    print(f"noisyhate test: {len(rows)} kalimat | slot tidak sejajar: {dropped}")


if __name__ == "__main__":
    prepare_mln()
    prepare_conda()
    prepare_noisyhate()
