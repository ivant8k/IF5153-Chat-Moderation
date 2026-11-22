"""Predict Task 2: kontrak input-output untuk integrasi pipeline

from predict import ToxicityClassifier
clf = ToxicityClassifier("e2_mix", mode="balanced")      # run id di outputs/task_2/runs
clf.classify("you are idiot")      # -> dict
clf.classify(["u | r | idiot", "gg wp"])  # -> list dict (delimiter ' | ' otomatis dibuang)
"""
import argparse
import json

from common import clean_text, read_json
from config import CATS, RUNS, WARN_NAMES


class ToxicityClassifier:
    def __init__(self, run, mode="balanced", device=None):
        from model import TorchPredictor
        if mode not in ("balanced", "strict"):
            raise ValueError("mode harus 'balanced' atau 'strict'")
        self.mode = mode
        self.run_dir = RUNS / run
        self.th = read_json(self.run_dir / "thresholds.json")
        self.net = TorchPredictor(self.run_dir / "checkpoints" / "best", device)

    def classify(self, text):
        single = isinstance(text, str)
        original_texts = [text] if single else list(text)
        if not all(isinstance(t, str) for t in original_texts):
            raise TypeError("text harus berupa string atau list string")
        texts = [clean_text(t) for t in original_texts]
        probs = self.net.probs(texts)
        out = [self._result(original, p) for original, p in zip(original_texts, probs)]
        return out[0] if single else out

    def _result(self, text, p):
        gate_thr = self.th["gate"][self.mode]
        trigger = bool(text) and bool(p[0] >= gate_thr)
        labels = {c: round(float(p[1 + k]), 4) for k, c in enumerate(CATS)}
        warnings, message = [], None
        if trigger:
            specific = [c for c in CATS if c != "toxic" and labels[c] >= self.th["cats"][c]]
            specific.sort(key=lambda c: -labels[c])
            warnings = specific or ["toxic"]   # fallback: gate menyala tapi tidak ada kategori spesifik
            names = [WARN_NAMES[c] for c in warnings]
            joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            message = f"Warning: your message contains {joined}."
        return {"text": text, "p_toxic": round(float(p[0]), 4), "trigger": trigger, "labels": labels,
                "warnings": warnings, "message": message, "mode": self.mode, "threshold": round(gate_thr, 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", default="balanced")
    ap.add_argument("text", nargs="+")
    a = ap.parse_args()
    print(json.dumps(ToxicityClassifier(a.run, a.mode).classify(a.text), indent=2, ensure_ascii=False))
