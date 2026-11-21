"""Phase 5 verification: subset splitting, error categorization, test gold immutability, and output schema."""
import ast
import csv
import json
from pathlib import Path
import sys

if __package__:
    from ..config import DATA_FILES, RESULTS_DIR
    from ..dataset import load_semeval_split
    from ..evaluate import classify_error, split_gold_subset, summarize_predictions
else:
    TASK_DIR = str(Path(__file__).resolve().parents[1])
    if TASK_DIR not in sys.path:
        sys.path.insert(0, TASK_DIR)
    from config import DATA_FILES, RESULTS_DIR
    from dataset import load_semeval_split
    from evaluate import classify_error, split_gold_subset, summarize_predictions


def log(tag: str, message: str, *, err: bool | None = None) -> None:
    if err is None:
        err = tag in ("ERROR", "FATAL", "WARN")
    print(f"[{tag}] {message}", file=sys.stderr if err else sys.stdout)


def test_subset_and_metrics() -> None:
    """Verify empty/single/multi/long_phrase splitting and official SemEval F1 semantics."""
    assert split_gold_subset([], "clean chat") == ("empty", False)
    assert split_gold_subset([4, 5, 6, 7], "you fool") == ("single", False)
    assert split_gold_subset([0, 1, 2, 10, 11, 12], "bad word! bad") == ("multi", False)
    assert split_gold_subset(list(range(0, 21)), "very bad toxic phrase ok") == ("single", True)

    rows = [
        {"text": "clean", "spans": []},
        {"text": "you fool", "spans": [4, 5, 6, 7]},
        {"text": "bad and bad", "spans": [0, 1, 2, 8, 9, 10]},
    ]
    golds = [r["spans"] for r in rows]
    preds = [[], [4, 5, 6, 7], [0, 1, 2]]
    stat, scores = summarize_predictions(preds, golds, rows)
    assert scores[:2] == [1.0, 1.0] and abs(scores[2] - (2.0 / 3.0)) < 1e-9
    assert stat["subsets"]["empty"]["n"] == 1 and stat["subsets"]["empty"]["f1"] == 1.0
    assert stat["subsets"]["single"]["n"] == 1 and stat["subsets"]["single"]["f1"] == 1.0
    assert stat["subsets"]["multi"]["n"] == 1 and abs(stat["subsets"]["multi"]["f1"] - (2.0 / 3.0)) < 1e-9


def test_error_classification() -> None:
    """Verify mutually exclusive error categories."""
    assert classify_error([], [], 1.0) == "exact_match"
    assert classify_error([1, 2], [], 0.0) == "empty_false_positive"
    assert classify_error([], [1, 2], 0.0) == "missed_toxic"
    assert classify_error([5, 6], [1, 2], 0.0) == "zero_overlap_mismatch"
    assert classify_error([1, 2, 3], [1, 2], 0.8) == "partial_over_extension"
    assert classify_error([1], [1, 2], 0.66) == "partial_under_extension"


def test_test_gold_unmodified() -> None:
    """Verify load_semeval_split('test') returns raw tsd_test.csv spans without fix_spans."""
    loaded = load_semeval_split("test")
    with open(DATA_FILES["test"], encoding="utf-8", newline="") as f:
        raw = [{"text": r["text"], "spans": ast.literal_eval(r["spans"])} for r in csv.DictReader(f)]
    assert len(loaded) == len(raw) == 2000
    assert loaded == raw


def test_eval_outputs_if_present() -> None:
    """Verify eval_summary.json schema if evaluate.py has already been executed."""
    summary_path = RESULTS_DIR / "eval" / "eval_summary.json"
    if not summary_path.exists():
        log("SKIP", f"{summary_path} belum ada; jalankan python -m src.task_3.evaluate terlebih dahulu")
        return
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    for split in ("trial", "test"):
        assert split in data
        for m in ("lexicon", "bert_linear", "bert_crf_raw", "bert_crf_post"):
            assert m in data[split]
            assert "char" in data[split][m] and "subsets" in data[split][m]
    log("CHECKPOINT 4", "Schema eval_summary.json tervalidasi")


def main() -> int:
    log("START", "Memulai verifikasi modul evaluasi Phase 5")
    test_subset_and_metrics()
    log("CHECKPOINT 1", "Logika breakdown subset & metrik lulus")
    test_error_classification()
    log("CHECKPOINT 2", "Kategorisasi error analysis lulus")
    test_test_gold_unmodified()
    log("CHECKPOINT 3", "Imutabilitas gold tsd_test.csv (2000 baris) lulus")
    test_eval_outputs_if_present()
    log("DONE", "Seluruh verifikasi Phase 5 lulus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
