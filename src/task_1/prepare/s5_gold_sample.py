"""Gold Sample: conda bench -> gold_conda.tsv + gold_strata.json (stratified, sekali)"""
import random

from common import load_lexicon, read_tsv, split_slots, write_json, write_tsv
from config import ANN_LINE, GOLD_QUOTA, OUT_CONDA, SEED

GOLD_FIELDS = ["id", "stratum", "raw", "src", "silver", "gold", "checked", "note"]


def stratum_of(row, toxic, game):
    """row -> toxic, changed, game, atau other"""
    src, tgt = split_slots(row["src"]), split_slots(row["tgt"])
    checks = {
        "toxic": any(s in toxic or t in toxic for s, t in zip(src, tgt)),
        "changed": row["src"] != row["tgt"],
        "game": any(s in game for s in src),
        "other": True,
    }
    return next(name for name, _ in GOLD_QUOTA if checks[name])


def main():
    path = ANN_LINE / "gold_conda.tsv"
    if path.exists():
        print(f"{path.name} sudah ada, tidak ditimpa")
        return

    lex = load_lexicon()
    game = lex["C"] | lex["D"] | lex["S"]

    # Unique Lines
    seen, unique = set(), []
    for row in read_tsv(OUT_CONDA / "bench.tsv"):
        if row["src"] not in seen:
            seen.add(row["src"])
            unique.append(row)

    # Stratified Sampling
    strata = {name: [] for name, _ in GOLD_QUOTA}
    for row in unique:
        strata[stratum_of(row, lex["T"], game)].append(row)
    rng = random.Random(SEED)
    picked = []
    for name, quota in GOLD_QUOTA:
        for row in rng.sample(strata[name], min(quota, len(strata[name]))):
            picked.append({"id": row["id"], "stratum": name, "raw": row["raw"], "src": row["src"],
                           "silver": row["tgt"], "gold": row["tgt"], "checked": "", "note": ""})
    picked.sort(key=lambda r: ([n for n, _ in GOLD_QUOTA].index(r["stratum"]), r["id"]))
    write_tsv(path, picked, GOLD_FIELDS)

    population = {name: len(rows) for name, rows in strata.items()}
    write_json(ANN_LINE / "gold_strata.json", population)
    print(f"gold: {len(picked)} baris | populasi bench unik: {population}")


if __name__ == "__main__":
    main()
