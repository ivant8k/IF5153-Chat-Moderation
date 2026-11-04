"""Common Task 1: baca tulis TSV, tokenisasi, format slot, corong lapis"""
import csv
import re
from collections import Counter

from config import ANN, CONDA_DIR, DELIM, MIN_FREQ_VOCAB, MLN_DIR, PIPE_SUB, VOCAB_FILE

EDGE = re.compile(r"^(\W*)(.*?\w)(\W*)$")
LAUGH = re.compile(r"(?=.*a)(?=.*h)[ah]{3,}|(?=.*e)(?=.*h)[eh]{3,}|x+d+|l+o+l+|z{3,}")


# TSV
def read_tsv(path):
    """TSV polos tanpa quoting -> list dict"""
    lines = path.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split("\t")
    return [dict(zip(fields, line.split("\t"))) for line in lines[1:] if line]


def write_tsv(path, rows, fields):
    """List dict -> TSV polos, nilai tidak boleh berisi tab atau newline"""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = ["\t".join(fields)]
    for row in rows:
        values = [str(row.get(k, "")) for k in fields]
        assert not any(c in v for v in values for c in "\t\n\r"), row
        out.append("\t".join(values))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# Tokenization
def split_edges(chunk):
    """Chunk -> tanda baca depan, kata, tanda baca belakang. Simbol leet tetap menempel"""
    lead, word, tail = EDGE.match(chunk).groups()
    # $tupid, !diot, @ss: satu simbol leet tepat sebelum huruf ikut ke kata
    if lead and lead[-1] in "$@!":
        lead, word = lead[:-1], lead[-1] + word
    # jacka$$: $ di belakang ikut ke kata
    dollars = len(tail) - len(tail.lstrip("$"))
    word, tail = word + tail[:dollars], tail[dollars:]
    return [p for p in (lead, word, tail) if p]


def group_letters(tokens, min_run=3):
    """Run huruf tunggal (f u c k) -> satu token"""
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
    """Raw line -> list (token, start, end), tanda baca di tepi kata dipisah"""
    tokens = []
    for m in re.finditer(r"\S+", text):
        chunk, start = m.group(), m.start()
        parts = [chunk]
        # emoticon (:d, =), <3) dan simbol murni dibiarkan utuh
        if chunk[0] not in ":;=<" and any(c.isalnum() for c in chunk):
            parts = split_edges(chunk)
        for p in parts:
            tokens.append((p, start, start + len(p)))
            start += len(p)
    return group_letters(tokens)


def clean_slot(token):
    """Token -> isi slot: lowercase, '|' diganti"""
    return token.lower().replace("|", PIPE_SUB)


def to_src(text):
    """Raw line -> src ber-delimiter"""
    return DELIM.join(clean_slot(t) for t, _, _ in tokenize(text))


def split_slots(line):
    return line.split(DELIM)


# MultiLexNorm
def read_norm(path):
    """.norm -> list kalimat, tiap kalimat list (raw, norm)"""
    sents = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        pairs = []
        for line in block.splitlines():
            parts = line.split("\t")
            pairs.append((parts[0], parts[1] if len(parts) > 1 else ""))
        sents.append(pairs)
    return sents


def mln_slots(pairs):
    """(raw, norm) -> slot raw, slot target. Ekor merge digabung ke slot sebelumnya, run huruf tunggal digabung"""
    src, tgt = [], []
    for raw, norm in pairs:
        if norm == "" and src:
            src[-1] += " " + clean_slot(raw)
        else:
            src.append(clean_slot(raw))
            tgt.append(clean_slot(norm))
    # samakan dengan group_letters saat inferensi
    grouped = group_letters([(s, i, i) for i, s in enumerate(src)])
    new_src, new_tgt = [], []
    for slot, first, last in grouped:
        new_src.append(slot)
        new_tgt.append(" ".join(t for t in tgt[first:last + 1] if t))
    return new_src, new_tgt


def mln_table():
    """MLN train -> kamus raw ke norm paling sering, tanpa token yang terlibat merge"""
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


# Funnel Resources
def load_lexicon():
    """Lexicon CONDA -> dict slot (C, D, P, S, T) ke set token"""
    lex = {k: set() for k in "CDPST"}
    with open(CONDA_DIR / "resources" / "lexicon_refined_1209.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for k in lex:
                if row[k]:
                    lex[k].add(row[k].lower())
    return lex


def load_vocab():
    """Wortschatz rank, kata, frekuensi -> set kata lowercase yang cukup sering"""
    freq = Counter()
    for line in VOCAB_FILE.read_text(encoding="utf-8").splitlines():
        _, word, n = line.split("\t")
        freq[word.lower()] += int(n)
    return {w for w, n in freq.items() if n >= MIN_FREQ_VOCAB}


def load_whitelist():
    path = ANN / "whitelist.txt"
    if not path.exists():
        return set()
    return {w.strip().lower() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()}


def load_resources():
    lex = load_lexicon()
    return {
        "protected": lex["C"] | lex["D"] | lex["S"] | load_whitelist(),
        "toxic": lex["T"],
        "mln": mln_table(),
        "vocab": load_vocab(),
    }


# Funnel
def assign_layer(tok, res):
    """Token src -> (lapis, norm otomatis). Lapis 6 dan 7 belum punya norm"""
    if tok in res["protected"]:
        return 1, tok
    if tok in res["mln"]:
        norm = res["mln"][tok]
        return (2, norm) if norm != tok else (3, tok)
    if not any(c.isalpha() for c in tok) or tok[0] in ":;=" or LAUGH.fullmatch(tok):
        return 4, tok
    if tok in res["vocab"] and not any(c.isdigit() for c in tok):
        return 5, tok
    return None, None
