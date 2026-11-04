"""Candidates: conda lines -> source label per tipe, todo LLM, review MLN"""
from collections import Counter, defaultdict

from common import assign_source, load_resources, read_tsv, split_slots, write_tsv
from config import ANN_TYPE, MIN_FREQ_TYPE, OUT_CONDA, OUT_LABELING

SOURCES = ["context", "protected", "mln_changed", "mln_same", "symbol_laugh", "english", "llm_type", "llm_slot"]
REVIEW_FIELDS = ["raw", "mln_norm", "count", "ex1", "ex2", "final", "checked", "note"]


def collect(lines):
    """lines -> jumlah tipe, contoh baris"""
    counts, examples = Counter(), defaultdict(list)
    for line in lines:
        for tok in split_slots(line["src"]):
            counts[tok] += 1
            ex = examples[tok]
            if len(ex) < 3 and line["raw"] not in ex:
                ex.append(line["raw"])
    return counts, examples


def update_review(rows_new):
    """review lama + baris baru, isian lama dipertahankan"""
    path = ANN_TYPE / "review_mln.tsv"
    old = {r["raw"]: r for r in read_tsv(path)} if path.exists() else {}
    merged = []
    for r in rows_new:
        if r["raw"] in old:
            keep = old.pop(r["raw"])
            keep["count"] = r["count"]
            merged.append(keep)
        else:
            merged.append(r)
    merged.extend(old.values())
    write_tsv(path, merged, REVIEW_FIELDS)
    return len(merged)


def main():
    lines = read_tsv(OUT_CONDA / "lines.tsv")
    res = load_resources()
    counts, examples = collect(lines)

    # Type Sources
    types, source_of = [], {}
    for tok, n in counts.most_common():
        source, norm = assign_source(tok, res)
        if source is None:
            source = "llm_type" if n >= MIN_FREQ_TYPE else "llm_slot"
        source_of[tok] = source
        types.append({"raw": tok, "count": n, "source": source, "auto_norm": norm or ""})
    types.sort(key=lambda t: (SOURCES.index(t["source"]), -t["count"], t["raw"]))
    write_tsv(OUT_LABELING / "types.tsv", types, ["raw", "count", "source", "auto_norm"])

    # MLN Review Sheet
    review = []
    for t in types:
        # tipe context tetap masuk review
        if t["source"] == "mln_changed" or (t["source"] == "context" and t["raw"] in res["mln"]):
            ex = examples[t["raw"]] + ["", ""]
            norm = res["mln"][t["raw"]]
            review.append({"raw": t["raw"], "mln_norm": norm, "count": t["count"],
                           "ex1": ex[0], "ex2": ex[1], "final": norm, "checked": "", "note": ""})
    n_review = update_review(review)

    # LLM Type Todo
    todo_types = []
    for t in types:
        if t["source"] == "llm_type":
            ex = examples[t["raw"]] + ["", "", ""]
            todo_types.append({"raw": t["raw"], "count": t["count"], "ex1": ex[0], "ex2": ex[1], "ex3": ex[2]})
    write_tsv(OUT_LABELING / "todo_llm_types.tsv", todo_types, ["raw", "count", "ex1", "ex2", "ex3"])

    # LLM Slot Todo
    todo_slots = []
    for line in lines:
        for i, tok in enumerate(split_slots(line["src"])):
            if source_of[tok] in ("llm_slot", "context"):
                todo_slots.append({"line_id": line["id"], "split": line["split"], "slot": i,
                                   "raw": tok, "source": source_of[tok], "src": line["src"]})
    write_tsv(OUT_LABELING / "todo_llm_slots.tsv", todo_slots, ["line_id", "split", "slot", "raw", "source", "src"])

    # Summary
    total = sum(counts.values())
    print(f"tipe: {len(counts)} | token: {total}")
    for source in SOURCES:
        n_type = sum(1 for t in types if t["source"] == source)
        n_tok = sum(t["count"] for t in types if t["source"] == source)
        print(f"{source:13} tipe={n_type:6} token={n_tok:7} ({n_tok / total:.1%})")
    print(f"review_mln: {n_review} | todo_llm_types: {len(todo_types)} | todo_llm_slots: {len(todo_slots)}")


if __name__ == "__main__":
    main()
