"""Common: TSV, tokenisasi, slot, resource label"""
import csv
import json
import re
from collections import Counter

from config import ANN_RES, ANN_SLOT, CONDA_DIR, DELIM, MIN_FREQ_VOCAB, MLN_DIR, PIPE_SUB, VOCAB_FILE

EDGE = re.compile(r"^(\W*)(.*?\w)(\W*)$")
LAUGH = re.compile(r"(?=.*a)(?=.*h)[ah]{3,}|(?=.*e)(?=.*h)[eh]{3,}|x+d+|l+o+l+|z{3,}")


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


def write_json(path, data, sort_keys=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=sort_keys) + "\n", encoding="utf-8")


def read_list(path):
    """file satu kata per baris -> set"""
    if not path.exists():
        return set()
    return {w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()}


# Tokenization
def split_edges(chunk):
    """chunk -> tanda baca depan, kata, tanda baca belakang"""
    lead, word, tail = EDGE.match(chunk).groups()
    # $tupid, !diot, @ss
    if lead and lead[-1] in "$@!":
        lead, word = lead[:-1], lead[-1] + word
    # jacka$$
    dollars = len(tail) - len(tail.lstrip("$"))
    word, tail = word + tail[:dollars], tail[dollars:]
    return [p for p in (lead, word, tail) if p]


def group_letters(tokens, min_run=3):
    """f u c k -> satu token"""
    out, run = [], []
    for tok in tokens + [("", 0, 0)]:
        if len(tok[0]) == 1 and tok[0].isalpha():
            run.append(tok)
            continue
        if len(run) >= min_run:
            out.append((" ".join(t for t, _, _ in run), run[0][1], run[-1][2]))
        else:
            out.extend(run)
        run = []
        if tok[0]:
            out.append(tok)
    return out


def tokenize(text):
    """raw line -> list (token, start, end)"""
    tokens = []
    for m in re.finditer(r"\S+", text):
        chunk, start = m.group(), m.start()
        parts = [chunk]
        # emoticon (:d, =), <3) dan simbol murni utuh
        if chunk[0] not in ":;=<" and any(c.isalnum() for c in chunk):
            parts = split_edges(chunk)
        for p in parts:
            tokens.append((p, start, start + len(p)))
            start += len(p)
    return group_letters(tokens)


def clean_slot(token):
    """token -> isi slot lowercase"""
    return token.lower().replace("|", PIPE_SUB)


def to_src(text):
    """raw line -> src ber-delimiter"""
    return DELIM.join(clean_slot(t) for t, _, _ in tokenize(text))


def split_slots(line):
    return line.split(DELIM)


# MultiLexNorm
def read_norm(path):
    """.norm -> list kalimat (raw, norm)"""
    sents = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        pairs = []
        for line in block.splitlines():
            parts = line.split("\t")
            pairs.append((parts[0], parts[1] if len(parts) > 1 else ""))
        sents.append(pairs)
    return sents


def mln_slots(pairs):
    """(raw, norm) -> slot src, slot tgt"""
    src, tgt = [], []
    for raw, norm in pairs:
        # ekor merge masuk slot sebelumnya
        if norm == "" and src:
            src[-1] += " " + clean_slot(raw)
        else:
            src.append(clean_slot(raw))
            tgt.append(clean_slot(norm))
    grouped = group_letters([(s, i, i) for i, s in enumerate(src)])
    new_src, new_tgt = [], []
    for slot, first, last in grouped:
        new_src.append(slot)
        new_tgt.append(" ".join(t for t in tgt[first:last + 1] if t))
    return new_src, new_tgt


def mln_table():
    """MLN train -> raw: norm terbanyak"""
    counts = Counter()
    for pairs in read_norm(MLN_DIR / "en" / "train.norm"):
        for i, (raw, norm) in enumerate(pairs):
            in_merge = norm == "" or (i + 1 < len(pairs) and pairs[i + 1][1] == "")
            if not in_merge:
                counts[(raw.lower(), norm.lower())] += 1
    table = {}
    for (raw, norm), _ in counts.most_common():
        table.setdefault(raw, norm)
    return table


# Label Resources
def load_lexicon():
    """lexicon CONDA -> {C, D, P, S, T: set token}"""
    lex = {k: set() for k in "CDPST"}
    with open(CONDA_DIR / "resources" / "lexicon_refined_1209.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for k in lex:
                if row[k]:
                    lex[k].add(row[k].lower())
    return lex


def load_vocab():
    """Wortschatz -> set kata baku"""
    freq = Counter()
    for line in VOCAB_FILE.read_text(encoding="utf-8").splitlines():
        _, word, n = line.split("\t")
        freq[word.lower()] += int(n)
    return {w for w, n in freq.items() if n >= MIN_FREQ_VOCAB}


def load_protected(lex):
    return lex["C"] | lex["D"] | lex["S"] | read_list(ANN_RES / "whitelist.txt")


def load_resources():
    lex = load_lexicon()
    return {
        "protected": load_protected(lex),
        "toxic": lex["T"],
        "mln": mln_table(),
        "vocab": load_vocab(),
        "context": read_list(ANN_SLOT / "context_types.txt"),
    }


def is_toxic(norm, toxic):
    return any(w in toxic for w in norm.split())


# Label Source
def assign_source(tok, res):
    """token -> (source, norm otomatis)"""
    if tok in res["context"]:
        return "context", None
    if tok in res["protected"]:
        return "protected", tok
    if tok in res["mln"]:
        norm = res["mln"][tok]
        return ("mln_changed", norm) if norm != tok else ("mln_same", tok)
    if not any(c.isalpha() for c in tok) or tok[0] in ":;=" or LAUGH.fullmatch(tok):
        return "symbol_laugh", tok
    if tok in res["vocab"] and not any(c.isdigit() for c in tok):
        return "english", tok
    return None, None
