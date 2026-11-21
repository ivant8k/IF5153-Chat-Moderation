import numpy as np
import pandas as pd

import common
from config import CATS, LABELS
from predict import ToxicityClassifier


def test_build_train_masks_unlabeled_conda_categories(monkeypatch):
    jigsaw = pd.DataFrame([
        {"text": "jigsaw toxic", "gate": 1, **dict.fromkeys(CATS, 0)},
        {"text": "jigsaw clean", "gate": 0, **dict.fromkeys(CATS, 0)},
    ])
    jigsaw.loc[0, "toxic"] = 1
    conda = pd.DataFrame([
        {"norm": "explicit", "raw": "explicit raw", "gate": 1},
        {"norm": "action", "raw": "action raw", "gate": 0},
    ])
    monkeypatch.setattr(common, "load_set", lambda name: {"jigsaw_train": jigsaw, "conda_train": conda}[name])

    texts, targets, masks = common.build_train(["jigsaw", "conda"], ["norm"], 1.0, 7)

    assert texts == ["jigsaw toxic", "jigsaw clean", "explicit", "action"]
    assert targets.shape == masks.shape == (4, len(LABELS))
    np.testing.assert_array_equal(masks[:2], np.ones((2, len(LABELS))))
    np.testing.assert_array_equal(masks[2], [1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(masks[3], np.ones(len(LABELS)))
    np.testing.assert_array_equal(targets[2:], [[1, 0, 0, 0, 0, 0, 0], np.zeros(len(LABELS))])


def test_classifier_keeps_task1_delimiters_for_task3():
    class StubPredictor:
        def probs(self, texts):
            self.texts = texts
            return np.array([[0.9, 0.85, 0.01, 0.01, 0.01, 0.8, 0.01]])

    classifier = ToxicityClassifier.__new__(ToxicityClassifier)
    classifier.mode = "balanced"
    classifier.th = {"gate": {"balanced": 0.5}, "cats": dict.fromkeys(CATS, 0.5)}
    classifier.net = StubPredictor()

    result = classifier.classify("you | are | idiot")

    assert classifier.net.texts == ["you are idiot"]
    assert result["text"] == "you | are | idiot"
    assert result["trigger"] is True
    assert result["warnings"] == ["insult"]
