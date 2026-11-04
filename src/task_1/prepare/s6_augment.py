"""Augment: data bersih -> salinan dengan satu kata disamarkan (tgt tetap)"""
import random
from collections import Counter, defaultdict

from common import load_lexicon, read_tsv, split_slots, tokenize, write_json, write_tsv
from config import (ANN_RES, AUG_OPS, AUG_RATIO, DELIM, OUT_AUG, OUT_CONDA, OUT_MLN, OUT_REPORT, SEED,
                    TOXIC_WEIGHT)

AUG_FIELDS = ["id", "split", "source", "raw", "src", "tgt", "op", "slot"]
LEET = {"a": "4@", "e": "3", "i": "1!", "o": "0", "s": "5$", "t": "7", "l": "1", "g": "9", "b": "8"}
ACCENT = {"a": "áàä", "e": "éèë", "i": "íìï", "o": "óòö", "u": "úùü", "c": "ç", "n": "ñ", "y": "ý"}
VOWELS = "aeiou"


# Obfuscation Operators
def op_repeat(w, rng):
    i = rng.randrange(len(w))
    return w[:i + 1] + w[i] * rng.randint(2, 4) + w[i + 1:]


def op_leet(w, rng):
    pos = [i for i, c in enumerate(w) if c in LEET]
    if not pos:
        return None
    chars = list(w)
    for i in rng.sample(pos, max(1, len(pos) // 2)):
        chars[i] = rng.choice(LEET[chars[i]])
    return "".join(chars)


def op_symbol(w, rng):
    style = rng.choice(["mask", "insert", "between"])
    inner = [i for i in range(1, len(w) - 1) if w[i] in VOWELS]
    if style == "mask" and inner:
        i = rng.choice(inner)
        return w[:i] + "*" + w[i + 1:]
    if style == "between":
        return rng.choice(".-*").join(w)
    i = rng.randrange(1, len(w))
    return w[:i] + rng.choice("*.-") + w[i:]


def op_drop(w, rng):
    if len(w) < 4:
        return None
    i = rng.randrange(1, len(w) - 1)
    return w[:i] + w[i + 1:]


def op_accent(w, rng):
    pos = [i for i, c in enumerate(w) if c in ACCENT]
    if not pos:
        return None
    i = rng.choice(pos)
    return w[:i] + rng.choice(ACCENT[w[i]]) + w[i + 1:]


def op_spaced(w, rng):
    return " ".join(w) if len(w) <= 8 else None


def op_leet_repeat(w, rng):
    return op_leet(op_repeat(w, rng), rng)


OPS = {"repeat": op_repeat, "leet": op_leet, "symbol": op_symbol, "drop": op_drop,
       "accent": op_accent, "spaced": op_spaced, "leet_repeat": op_leet_repeat}


# Slot Selection
def is_valid(word, obf):
    """hasil samaran tetap satu slot di tokenizer"""
    if not obf or obf == word or "|" in obf:
        return False
    toks = tokenize(obf)
    return len(toks) == 1 and toks[0][0] == obf


def eligible_slots(row):
    """slot bersih, satu kata alfabet, panjang >= 3"""
    out = []
    for i, (s, t) in enumerate(zip(split_slots(row["src"]), split_slots(row["tgt"]))):
        if s == t and t.isascii() and t.isalpha() and len(t) >= 3:
            out.append(i)
    return out


def pick_slot(row, slots, toxic, rng):
    tgt = split_slots(row["tgt"])
    weights = [TOXIC_WEIGHT if tgt[i] in toxic else 1 for i in slots]
    return rng.choices(slots, weights)[0]


def obfuscate(word, op, real_forms, rng):
    """kata -> (samaran, op)"""
    # bentuk nyata dari lexicon, separuh waktu
    if word in real_forms and rng.random() < 0.5:
        return rng.choice(real_forms[word]), "real"
    obf = OPS[op](word, rng)
    return (obf, op) if is_valid(word, obf) else (None, op)


def augment_row(row, i, obf, op, split):
    src = split_slots(row["src"])
    src[i] = obf
    return {"id": f"{row['id']}-aug", "split": split, "source": f"{row['source']}_aug",
            "raw": " ".join(src), "src": DELIM.join(src), "tgt": row["tgt"], "op": op, "slot": i}


def main():
    rng = random.Random(SEED)
    toxic = load_lexicon()["T"]
    real_forms = defaultdict(list)
    for r in read_tsv(ANN_RES / "obfuscation_lexicon.tsv"):
        if r["final"] != r["raw"]:
            real_forms[r["final"]].append(r["raw"])
    op_names = [name for name, _ in AUG_OPS]
    op_weights = [w for _, w in AUG_OPS]
    stats = {}

    # Train Augmentation
    train = read_tsv(OUT_MLN / "train.tsv") + read_tsv(OUT_CONDA / "train.tsv")
    out = []
    for row in train:
        slots = eligible_slots(row)
        if not slots or rng.random() >= AUG_RATIO:
            continue
        i = pick_slot(row, slots, toxic, rng)
        word = split_slots(row["tgt"])[i]
        # coba op lain jika gagal
        for _ in range(3):
            obf, op = obfuscate(word, rng.choices(op_names, op_weights)[0], real_forms, rng)
            if obf:
                out.append(augment_row(row, i, obf, op, "train"))
                break
    write_tsv(OUT_AUG / "train.tsv", out, AUG_FIELDS)
    stats["train"] = dict(Counter(r["op"] for r in out))

    # Evaluation Augmentation
    for split in ["dev", "bench"]:
        out, k = [], 0
        for row in read_tsv(OUT_CONDA / f"{split}.tsv"):
            slots = eligible_slots(row)
            if not slots:
                continue
            i = pick_slot(row, slots, toxic, rng)
            word = split_slots(row["tgt"])[i]
            # op bergiliran
            for _ in range(len(op_names)):
                op = op_names[k % len(op_names)]
                k += 1
                obf = OPS[op](word, rng)
                if is_valid(word, obf):
                    out.append(augment_row(row, i, obf, op, split))
                    break
        write_tsv(OUT_AUG / f"conda_{split}.tsv", out, AUG_FIELDS)
        stats[split] = dict(Counter(r["op"] for r in out))

    write_json(OUT_REPORT / "aug_stats.json", stats, sort_keys=True)
    print(stats)


if __name__ == "__main__":
    main()
