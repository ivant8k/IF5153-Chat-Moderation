"""Build: conda lines + annotations -> conda train/dev/bench + reports"""
from collections import Counter

from common import is_toxic, load_lexicon, load_protected, read_tsv, split_slots, write_json, write_tsv
from config import ANN_LINE, ANN_SLOT, ANN_TYPE, DELIM, OUT_CONDA, OUT_LABELING, OUT_REPORT, OUT_RES, SPLIT_RATIO

OUT_FIELDS = ["id", "split", "source", "raw", "src", "tgt"]


def read_optional(path):
    return read_tsv(path) if path.exists() else []


def load_labels():
    types = {r["raw"]: r for r in read_tsv(OUT_LABELING / "types.tsv")}
    review = {r["raw"]: r["final"] for r in read_tsv(ANN_TYPE / "review_mln.tsv")}
    llm_types = {r["raw"]: r["final"] for r in read_tsv(ANN_TYPE / "llm_types.tsv")}
    slots = {(r["line_id"], int(r["slot"])): r["norm"] for r in read_tsv(ANN_SLOT / "llm_slots.tsv")}
    overrides = {r["id"]: r["tgt"] for r in read_optional(ANN_LINE / "line_overrides.tsv")}
    gold = {r["id"]: r["gold"] for r in read_optional(ANN_LINE / "gold_conda.tsv") if r["checked"] == "1"}
    return types, review, llm_types, slots, overrides, gold


def label_line(line, types, review, llm_types, slots):
    """line -> list (norm, origin)"""
    out = []
    for i, tok in enumerate(split_slots(line["src"])):
        source = types[tok]["source"]
        if (line["id"], i) in slots:
            out.append((slots[(line["id"], i)], "llm_slot"))
        elif source == "mln_changed":
            out.append((review[tok], "mln_review"))
        elif source == "llm_type":
            out.append((llm_types[tok], "llm_type"))
        elif source in ("llm_slot", "context"):
            out.append((None, "missing"))
        else:
            out.append((types[tok]["auto_norm"], "auto"))
    return out


def validate(src, tgt, protected, toxic, semantic=True):
    """src, tgt -> alasan gagal ('' jika lolos)"""
    src_slots, tgt_slots = split_slots(src), split_slots(tgt)
    if len(src_slots) != len(tgt_slots):
        return "slot_count"
    for s, t in zip(src_slots, tgt_slots):
        if not t.strip() or "|" in t or t != t.lower():
            return "format"
        if not semantic:
            continue
        if s in protected and t != s:
            return "protected_changed"
        if s in toxic and not is_toxic(t, toxic):
            return "toxic_lost"
    return ""


def make_whitelist(protected):
    """istilah game + kata MLN yang ditolak saat review"""
    rejected = {r["raw"] for r in read_tsv(ANN_TYPE / "review_mln.tsv")
                if r["final"] == r["raw"] and r["mln_norm"] != r["raw"]}
    words = sorted(protected | rejected)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    (OUT_RES / "whitelist.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
    return len(words)


def main():
    lines = read_tsv(OUT_CONDA / "lines.tsv")
    types, review, llm_types, slots, overrides, gold = load_labels()
    lex = load_lexicon()
    protected = load_protected(lex)
    toxic = lex["T"]

    # Label Merge
    outputs = {split: [] for split, _ in SPLIT_RATIO}
    failures, origin = [], Counter()
    gold_total = gold_line_ok = gold_slot_ok = gold_slots = 0
    for line in lines:
        labeled = label_line(line, types, review, llm_types, slots)
        if any(norm is None for norm, _ in labeled):
            failures.append({**line, "tgt": "", "reason": "unlabeled"})
            continue
        silver = DELIM.join(norm for norm, _ in labeled)
        tgt = silver
        if line["id"] in gold:
            tgt = gold[line["id"]]
            origin["gold"] += len(labeled)
            gold_total += 1
            gold_line_ok += silver == tgt
            for a, b in zip(split_slots(silver), split_slots(tgt)):
                gold_slots += 1
                gold_slot_ok += a == b
        elif line["id"] in overrides:
            tgt = overrides[line["id"]]
            origin["line_override"] += len(labeled)
        else:
            origin.update(how for _, how in labeled)

        # Validation
        # label human cukup dicek strukturnya
        by_human = line["id"] in gold or line["id"] in overrides
        reason = validate(line["src"], tgt, protected, toxic, semantic=not by_human)
        if reason:
            failures.append({**line, "tgt": tgt, "reason": reason})
            continue
        outputs[line["split"]].append({**line, "source": "conda", "tgt": tgt})

    # Output Files
    for split, rows in outputs.items():
        write_tsv(OUT_CONDA / f"{split}.tsv", rows, OUT_FIELDS)
    write_tsv(OUT_REPORT / "failures.tsv", failures, ["id", "split", "raw", "src", "tgt", "reason"])
    n_white = make_whitelist(protected)

    stats = {
        "conda_lines": {s: len(rows) for s, rows in outputs.items()},
        "conda_changed_lines": {s: sum(r["src"] != r["tgt"] for r in rows) for s, rows in outputs.items()},
        "label_origin_slots": dict(origin),
        "failures": dict(Counter(f["reason"] for f in failures)),
        "whitelist_size": n_white,
        "gold": {
            "lines": gold_total,
            "silver_line_agreement": round(gold_line_ok / gold_total, 4) if gold_total else None,
            "silver_slot_agreement": round(gold_slot_ok / gold_slots, 4) if gold_slots else None,
        },
    }
    write_json(OUT_REPORT / "build_stats.json", stats, sort_keys=True)
    print(stats)


if __name__ == "__main__":
    main()
