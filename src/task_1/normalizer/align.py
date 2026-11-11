"""Align: output model -> slot prediksi"""
from common import split_slots


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_cost(a, b):
    return levenshtein(a, b) / max(len(a), len(b), 1)


def dtw_align(src_slots, words):
    """slot raw + kata normalized -> slot prediksi"""
    n, m = len(src_slots), len(words)
    if m == 0:
        return list(src_slots)
    inf = float("inf")
    dist = [[inf] * (m + 1) for _ in range(n + 1)]
    dist[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = min(dist[i - 1][j - 1], dist[i][j - 1], dist[i - 1][j])
            dist[i][j] = word_cost(src_slots[i - 1], words[j - 1]) + best

    # Backtrack
    owner = [n - 1] * m
    i, j = n, m
    while i > 0 and j > 0:
        owner[j - 1] = i - 1
        moves = [(dist[i - 1][j - 1], i - 1, j - 1), (dist[i][j - 1], i, j - 1), (dist[i - 1][j], i - 1, j)]
        _, i, j = min(moves)

    slots = [[] for _ in range(n)]
    for j, i in enumerate(owner):
        slots[i].append(words[j])
    # slot tanpa kata kembali ke raw
    return [" ".join(w) if w else src_slots[i] for i, w in enumerate(slots)]


def clean_output(text):
    return " ".join(text.lower().split())


def postprocess(src, out, fmt, protected=None, repair=False):
    """output model -> (slot prediksi, valid)"""
    src_slots = split_slots(src)
    text = clean_output(out)
    if fmt == "plain":
        slots, valid = dtw_align(src_slots, text.split()), True
    else:
        slots = [s.strip() for s in text.split("|")]
        valid = len(slots) == len(src_slots)
        if not valid:
            slots = dtw_align(src_slots, text.replace("|", " ").split()) if repair else list(src_slots)
    if repair:
        slots = [s if "<unk>" in p else p for s, p in zip(src_slots, slots)]
    if protected:
        slots = [s if s in protected else p for s, p in zip(src_slots, slots)]
    return slots, valid
