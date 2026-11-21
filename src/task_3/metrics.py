"""Task 3 metrics: wrapper over official SemEval-2021 F1 and seqeval."""
import sys

import numpy as np
from scipy.stats import sem
from seqeval.metrics import f1_score as seqeval_f1
from seqeval.metrics import precision_score as seqeval_precision
from seqeval.metrics import recall_score as seqeval_recall

if __package__:
    from .config import EVAL_DIR
else:
    from config import EVAL_DIR

if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from semeval2021 import f1 as semeval_f1


def evaluate_char_spans(preds: list[list[int]], golds: list[list[int]]) -> dict[str, float | int]:
    """Compute official SemEval-2021 Character F1 overall and by empty/non-empty gold subsets."""
    if len(preds) != len(golds):
        raise ValueError(f"Length mismatch: preds={len(preds)} vs golds={len(golds)}")

    scores = [semeval_f1(pred, gold) for pred, gold in zip(preds, golds)]
    empty_scores = [s for s, g in zip(scores, golds) if not g]
    non_empty_scores = [s for s, g in zip(scores, golds) if g]

    return {
        "n": len(scores),
        "f1": float(np.mean(scores)) if scores else 0.0,
        "sem": float(sem(scores)) if len(scores) > 1 else 0.0,
        "empty_n": len(empty_scores),
        "empty_f1": float(np.mean(empty_scores)) if empty_scores else 0.0,
        "non_empty_n": len(non_empty_scores),
        "non_empty_f1": float(np.mean(non_empty_scores)) if non_empty_scores else 0.0,
    }


def evaluate_bio_tags(pred_tags: list[list[str]], gold_tags: list[list[str]]) -> dict[str, float]:
    """Compute span-level BIO precision, recall, and F1 via seqeval."""
    if len(pred_tags) != len(gold_tags):
        raise ValueError(f"Length mismatch: pred_tags={len(pred_tags)} vs gold_tags={len(gold_tags)}")

    return {
        "precision": float(seqeval_precision(gold_tags, pred_tags, zero_division=0)),
        "recall": float(seqeval_recall(gold_tags, pred_tags, zero_division=0)),
        "f1": float(seqeval_f1(gold_tags, pred_tags, zero_division=0)),
    }
