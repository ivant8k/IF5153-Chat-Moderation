import csv
import re
from collections import defaultdict

from loader import task_scope
from settings import DELIM, ROOT, SEPA, T1_PREPARE_DIR

T1_DATA = ROOT / "data" / "processed" / "task_1"
CONDA_CSV = ROOT / "data" / "raw" / "conda" / "data"
GOLD_FILE = ROOT / "data" / "annotations" / "task_1" / "line_labels" / "gold_conda.tsv"
ID_RE = re.compile(r"^conda-(\d+)-(\d+)$")
SLOT_RE = re.compile(r"(.*?) \((\w+)\), ")
TOXIC_INTENTS = {"E", "I"}
GAME_SLOTS = {"C", "D", "S"}

with task_scope(T1_PREPARE_DIR):
    import common as t1_common
    tokenize = t1_common.tokenize


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def conda_labels():
    """CSV train + valid -> id: (intent, slot token per baris chat)"""
    out = {}
    for name in ["train", "valid"]:
        with open(CONDA_CSV / f"CONDA_{name}.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if not r["utterance"]:
                    continue
                lines, cur = [], []
                for tok, lab in SLOT_RE.findall(r["slotTokens"]):
                    if lab == "SEPA":
                        lines.append(cur)
                        cur = []
                    else:
                        cur.append((tok, lab))
                lines.append(cur)
                out[r["Id"]] = (r["intentClass"], lines)
    return out


def slot_labels(line, conda_tokens):
    """baris chat + token CONDA berlabel -> set label per slot Task 1 (None jika gagal dipetakan)"""
    spans, cursor, low, missed = [], 0, line.lower(), 0
    for tok, lab in conda_tokens:
        pos = low.find(tok.lower(), cursor)
        if pos < 0:
            # kontraksi diekspansi CONDA (you're -> you are), dilewati
            if lab == "T":
                return None
            missed += 1
            continue
        spans.append((pos, pos + len(tok), lab))
        cursor = pos + len(tok)
    if missed * 2 > len(conda_tokens):
        return None
    labels = []
    for _, start, end in tokenize(line):
        labels.append({lab for s, e, lab in spans if s < end and e > start})
    return labels


def row_item(rid, intent, parts, conda_lines, norm=None):
    """bagian baris -> item uji"""
    lines = [raw for _, raw in parts]
    labels = []
    for k, raw in parts:
        lab = slot_labels(raw, conda_lines[k]) if k < len(conda_lines) else None
        if lab is None:
            labels = None
            break
        labels += lab
    return {"id": rid, "intent": intent, "gate": int(intent in TOXIC_INTENTS), "lines": lines,
            "slot_labels": labels, "norm": norm}


def conda_bench():
    """bench Task 1 per baris CSV (chat [SEPA] digabung), norm = target silver Task 1"""
    labels = conda_labels()
    parts = defaultdict(list)
    for r in read_tsv(T1_DATA / "conda" / "bench.tsv"):
        rid, k = ID_RE.match(r["id"]).groups()
        parts[rid].append((int(k), r["raw"], r["tgt"].split(DELIM)))
    items = []
    for rid in sorted(parts, key=int):
        if rid not in labels:
            continue
        intent, conda_lines = labels[rid]
        rows = sorted(parts[rid])
        silver = [s for _, _, tgt in rows for s in tgt]
        items.append(row_item(f"conda-{rid}", intent, [(k, raw) for k, raw, _ in rows], conda_lines, silver))
    return items


def gold_conda():
    """300 baris chat bergold normalisasi, norm = slot gold (oracle)"""
    labels = conda_labels()
    items = []
    for r in read_tsv(GOLD_FILE):
        rid, k = ID_RE.match(r["id"]).groups()
        intent, conda_lines = labels[rid]
        item = row_item(r["id"], intent, [(int(k), r["raw"])], conda_lines, norm=r["gold"].split(DELIM))
        items.append(item)
    return items


def noisyhate():
    """kalimat tersamar, semua toksik, norm = versi bersih (oracle), slot berubah = kata tersamar"""
    items = []
    for r in read_tsv(T1_DATA / "noisyhate" / "test.tsv"):
        src, tgt = r["src"].split(DELIM), r["tgt"].split(DELIM)
        items.append({"id": r["id"], "intent": "E", "gate": 1, "lines": [r["raw"]], "slot_labels": None,
                      "norm": tgt, "perturbed": [i for i, (a, b) in enumerate(zip(src, tgt)) if a != b]})
    return items


SETS = {"conda_bench": conda_bench, "gold_conda": gold_conda, "noisyhate": noisyhate}

