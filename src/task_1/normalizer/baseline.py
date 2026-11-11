"""Baseline: LAI, MFR-MLN, MFR-all, RULE -> file prediksi format run"""
import argparse
import re
import unicodedata
from collections import Counter

from common import (load_set, load_toxic, load_vocab, pred_tag, read_tsv, set_names, split_slots, write_json,
                    write_preds)
from config import DATA, DELIM, SMOKE_EVAL_ROWS, TRAIN_FILES
from runs import BASELINES, run_dir

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g",
                      "@": "a", "$": "s", "!": "i"})


def mfr_table(parts):
    """data latih -> slot src: slot tgt terbanyak"""
    counts = Counter()
    for part in parts:
        for r in read_tsv(DATA / TRAIN_FILES[part]):
            counts.update(zip(split_slots(r["src"]), split_slots(r["tgt"])))
    table = {}
    for (src, tgt), _ in counts.most_common():
        table.setdefault(src, tgt)
    return table


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def rule_slot(slot, table, vocab):
    """slot -> normalisasi aturan, fallback MFR-all"""
    if slot in table:
        return table[slot]
    if slot in vocab or not any(c.isalpha() for c in slot):
        return slot
    base = strip_accents(slot.replace(" ", ""))
    leet = base.translate(LEET)
    cands = [base, leet, re.sub(r"[^a-z]", "", leet)]
    for c in list(cands):
        cands += [re.sub(r"(.)\1{2,}", r"\1\1", c), re.sub(r"(.)\1+", r"\1", c)]
    for c in cands:
        if c in vocab:
            return table.get(c, c)
    return slot


def make_baselines():
    """nama baseline -> fungsi slot -> slot"""
    mln, full = mfr_table(["mln"]), mfr_table(["mln", "conda"])
    vocab = load_vocab() | load_toxic()
    return {
        "base_lai": lambda s: s,
        "base_mfr-mln": lambda s: mln.get(s, s),
        "base_mfr-all": lambda s: full.get(s, s),
        "base_rule": lambda s: rule_slot(s, full, vocab),
    }


def run_baselines(smoke=False):
    funcs = make_baselines()
    for run_id in BASELINES:
        out_dir = run_dir(run_id, smoke)
        write_json(out_dir / "config.json", {"id": run_id, "kind": "baseline", "fmt": "delim"})
        for split in ("dev", "test"):
            for name in set_names(split):
                rows = load_set(name)[:SMOKE_EVAL_ROWS] if smoke else load_set(name)
                outs = [DELIM.join(funcs[run_id](s) for s in split_slots(r["src"])) for r in rows]
                write_preds(out_dir / "pred" / pred_tag(split, 1) / f"{name}.tsv", rows, outs)
        print(f"{run_id} selesai")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run_baselines(parser.parse_args().smoke)


if __name__ == "__main__":
    main()
