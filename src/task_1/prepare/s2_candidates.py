"""Candidates Task 1: conda_lines -> corong lapis 1-5 + daftar kerja lapis 6-7

Output:
  processed/types.tsv          : raw, count, layer, auto_norm
  processed/todo_llm_types.tsv : lapis 6, dilabel per tipe
  processed/todo_llm_slots.tsv : lapis 7, dilabel per slot dengan konteks baris
  annotations/review_mln.tsv   : lapis 2 untuk QC, isian lama tidak ditimpa
"""
from collections import Counter, defaultdict

from common import assign_layer, load_resources, read_tsv, split_slots, write_tsv
from config import ANN, MIN_FREQ_TYPE, OUT

REVIEW_FIELDS = ["raw", "mln_norm", "count", "ex1", "ex2", "final", "checked", "note"]


def collect(lines):
    """Hitung tipe + simpan contoh baris unik per tipe"""
    counts, examples = Counter(), defaultdict(list)
    for line in lines:
        for tok in split_slots(line["src"]):
            counts[tok] += 1
            ex = examples[tok]
            if len(ex) < 3 and line["raw"] not in ex:
                ex.append(line["raw"])
    return counts, examples


def update_review(rows_new):
    """Gabung review lama + baris baru, keputusan user dipertahankan"""
    path = ANN / "review_mln.tsv"
    old = {r["raw"]: r for r in read_tsv(path)} if path.exists() else {}
    merged = []
    for r in rows_new:
        if r["raw"] in old:
            keep = old.pop(r["raw"])
            keep["count"] = r["count"]
            merged.append(keep)
        else:
            merged.append(r)
    # tipe yang hilang dari data tetap disimpan
    merged.extend(old.values())
    write_tsv(path, merged, REVIEW_FIELDS)
    return len(merged)


def main():
    lines = read_tsv(OUT / "conda_lines.tsv")
    res = load_resources()
    counts, examples = collect(lines)

    # Layer Assignment
    types, layer_of = [], {}
    for tok, n in counts.most_common():
        layer, norm = assign_layer(tok, res)
        if layer is None:
            layer = 6 if n >= MIN_FREQ_TYPE else 7
        layer_of[tok] = layer
        types.append({"raw": tok, "count": n, "layer": layer, "auto_norm": norm or ""})
    types.sort(key=lambda t: (t["layer"], -t["count"], t["raw"]))
    write_tsv(OUT / "types.tsv", types, ["raw", "count", "layer", "auto_norm"])

    # Review Lapis 2
    review = []
    for t in types:
        if t["layer"] == 2:
            ex = examples[t["raw"]] + ["", ""]
            review.append({"raw": t["raw"], "mln_norm": t["auto_norm"], "count": t["count"],
                           "ex1": ex[0], "ex2": ex[1], "final": t["auto_norm"], "checked": "", "note": ""})
    n_review = update_review(review)

    # Todo Lapis 6
    todo_types = []
    for t in types:
        if t["layer"] == 6:
            ex = examples[t["raw"]] + ["", "", ""]
            todo_types.append({"raw": t["raw"], "count": t["count"], "ex1": ex[0], "ex2": ex[1], "ex3": ex[2]})
    write_tsv(OUT / "todo_llm_types.tsv", todo_types, ["raw", "count", "ex1", "ex2", "ex3"])

    # Todo Lapis 7
    todo_slots = []
    for line in lines:
        for i, tok in enumerate(split_slots(line["src"])):
            if layer_of[tok] == 7:
                todo_slots.append({"line_id": line["id"], "split": line["split"], "slot": i,
                                   "raw": tok, "src": line["src"]})
    write_tsv(OUT / "todo_llm_slots.tsv", todo_slots, ["line_id", "split", "slot", "raw", "src"])

    # Summary
    total = sum(counts.values())
    print(f"tipe: {len(counts)} | token: {total}")
    for layer in range(1, 8):
        n_type = sum(1 for t in types if t["layer"] == layer)
        n_tok = sum(t["count"] for t in types if t["layer"] == layer)
        print(f"lapis {layer}: tipe={n_type:6} token={n_tok:7} ({n_tok / total:.1%})")
    print(f"review_mln: {n_review} baris | todo lapis 6: {len(todo_types)} tipe | todo lapis 7: {len(todo_slots)} slot")


if __name__ == "__main__":
    main()
