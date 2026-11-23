import json
from collections import Counter, defaultdict

from scores import word_group
from sets import SETS, TOXIC_INTENTS, tokenize
from settings import DELIM, OUT

N_EXAMPLES = 8
# kata yang tidak boleh disensor, per jenis
OVER_CATEGORY = {"t_nontoxic": "masked_t_nontoxic", "game": "masked_game", "other": "masked_other"}


def load_preds(set_name, variant):
    path = OUT / "pred" / set_name / f"{variant}.jsonl"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def gate_errors(items, full, oracle, no_t1):
    """baris salah warning -> (kategori, penyebab)"""
    out = []
    for it in items:
        p = full[it["id"]]
        if it["gate"] and not p["trigger"]:
            cause = "T1" if oracle and oracle[it["id"]]["trigger"] else "T2"
            out.append((it, f"warn_miss_{it['intent']}", cause))
        elif not it["gate"] and p["trigger"]:
            cause = "T1" if no_t1 and not no_t1[it["id"]]["trigger"] else "T2"
            out.append((it, f"warn_false_{it['intent']}", cause))
    return out


def slot_errors(items, full, oracle, no_gate):
    """kata salah sensor -> (item, indeks slot, kategori, penyebab)"""
    out = []
    for it in items:
        if it["slot_labels"] is None:
            continue
        p = full[it["id"]]
        masked = set(p["masked_slots"])
        o_masked = set(oracle[it["id"]]["masked_slots"]) if oracle else set()
        g_masked = set(no_gate[it["id"]]["masked_slots"]) if no_gate else set()
        for i, lab in enumerate(it["slot_labels"]):
            group = word_group(lab, it["intent"] in TOXIC_INTENTS)
            if group == "target" and i not in masked:
                if not p["trigger"]:
                    cause = "T2" if i in g_masked else "T2+T3"
                else:
                    cause = "T1" if i in o_masked else "T3"
                out.append((it, i, "missed_target", cause))
            elif group != "target" and i in masked:
                if oracle and i not in o_masked:
                    cause = "T1"
                elif not it["gate"]:
                    # baris tidak toksik: warning T2 salah, lalu T3 memilih kata ini
                    cause = "T2+T3"
                else:
                    cause = "T3"
                out.append((it, i, OVER_CATEGORY[group], cause))
    return out


def example(it, pred, slot=None):
    row = {"id": it["id"], "intent": it["intent"], "raw": pred["raw"], "norm": pred["norm"],
           "p_toxic": pred["p_toxic"], "censored": pred["censored"]}
    if slot is not None:
        row["slot"] = pred["src"].split(DELIM)[slot]
    return row


def data_examples(items):
    """contoh baris berlabel dari data (tanpa prediksi): E dengan T, I tanpa T, A dan O dengan istilah game, O dengan T"""
    want = [("E", "target"), ("I", None), ("A", "game"), ("O", "game"), ("O", "t_nontoxic")]
    out = []
    for intent, group in want:
        for it in items:
            labs = it["slot_labels"]
            if it["intent"] != intent or labs is None or not 4 <= len(labs) <= 10 or len(it["lines"]) > 1:
                continue
            groups = [word_group(lab, intent in TOXIC_INTENTS) for lab in labs]
            if (group is None and "target" in groups) or (group and group not in groups):
                continue
            words = [w for w, _, _ in tokenize(it["lines"][0])]
            out.append({"intent": intent, "raw": it["lines"][0],
                        "words": [[w, "".join(sorted(lab)) or "O"] for w, lab in zip(words, labs)]})
            break
    return out


def analyze(set_name):
    items = SETS[set_name]()
    full = load_preds(set_name, "full")
    if full is None:
        return None
    oracle, no_t1, no_gate = (load_preds(set_name, v) for v in ["oracle_t1", "no_t1", "no_gate"])

    counts = Counter()
    examples = defaultdict(list)
    words = defaultdict(Counter)
    for it, cat, cause in gate_errors(items, full, oracle, no_t1):
        counts[(cat, cause)] += 1
        if len(examples[(cat, cause)]) < N_EXAMPLES:
            examples[(cat, cause)].append(example(it, full[it["id"]]))
    for it, i, cat, cause in slot_errors(items, full, oracle, no_gate):
        counts[(cat, cause)] += 1
        words[cat][full[it["id"]]["src"].split(DELIM)[i]] += 1
        if len(examples[(cat, cause)]) < N_EXAMPLES:
            examples[(cat, cause)].append(example(it, full[it["id"]], i))

    # Implicit Rows Without Target Word
    implicit = [it for it in items if it["intent"] == "I" and it["slot_labels"] is not None]
    no_word = [it for it in implicit if not any("T" in s for s in it["slot_labels"])]
    extra = {"implicit_rows": len(implicit), "implicit_without_target": len(no_word),
             "implicit_without_target_warned": sum(full[it["id"]]["trigger"] for it in no_word)}
    if set_name != "noisyhate":
        extra["data_examples"] = data_examples(items)

    # NoisyHate Perturbed Words
    if set_name == "noisyhate":
        for it in items:
            p = full[it["id"]]
            norm = p["norm"].split(DELIM)
            for i in it["perturbed"]:
                recovered = i < len(norm) and norm[i] == it["norm"][i]
                masked = i in p["masked_slots"]
                counts[("perturbed_word", f"{'recovered' if recovered else 'not_recovered'}_"
                        f"{'masked' if masked else 'unmasked'}")] += 1
    return counts, examples, words, extra


def run_analysis():
    """prediksi tersimpan -> outputs/pipeline/analysis/{summary.json, <set>_examples.jsonl}"""
    folder = OUT / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {}
    for set_name in SETS:
        res = analyze(set_name)
        if res is None:
            continue
        counts, examples, words, extra = res
        summary[set_name] = {"errors": [{"category": c, "cause": k, "count": n}
                                        for (c, k), n in sorted(counts.items())], **extra,
                             "top_words": {c: w.most_common(20) for c, w in words.items()}}
        with open(folder / f"{set_name}_examples.jsonl", "w", encoding="utf-8") as f:
            for (cat, cause), rows in sorted(examples.items()):
                for r in rows:
                    f.write(json.dumps({"category": cat, "cause": cause, **r}, ensure_ascii=False) + "\n")
    (folder / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
