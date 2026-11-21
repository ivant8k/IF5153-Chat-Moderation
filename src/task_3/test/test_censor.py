"""Phase 6 quality gate: RS-03/RC-03 delimiter alignment unit suite & BertCRF integration check."""
from pathlib import Path
import sys

if __package__:
    from ..censor import align_and_mask, task1_tokenize
    from ..config import RUNS_DIR
    from ..predict import ToxicSpanCensor
else:
    TASK_DIR = str(Path(__file__).resolve().parents[1])
    if TASK_DIR not in sys.path:
        sys.path.insert(0, TASK_DIR)
    from censor import align_and_mask, task1_tokenize
    from config import RUNS_DIR
    from predict import ToxicSpanCensor


def log(tag: str, message: str, *, err: bool | None = None) -> None:
    if err is None:
        err = tag in ("ERROR", "FATAL", "WARN")
    print(f"[{tag}] {message}", file=sys.stderr if err else sys.stdout)


def print_table(title: str, rows: list[list], line_width: int = 49) -> None:
    print(title)
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    total = sum(widths) + 3 * (len(widths) - 1)
    print("-" * max(line_width, total))
    for row in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())


def _stub_tagger(toxic_words: set[str]):
    def tagger(words: list[str]) -> list[str]:
        tags = []
        for w in words:
            if w.lower() in toxic_words:
                tags.append("I-TOXIC" if tags and tags[-1] != "O" else "B-TOXIC")
            else:
                tags.append("O")
        return tags

    return tagger


def test_five_rc03_scenarios() -> None:
    """Verify the 5 mandatory RS-03/RC-03 scenarios from GUIDE.md."""
    tagger = _stub_tagger({"fuck", "idiot", "bitch", "ass"})

    # 1. Token Expansion (RS-03)
    r1 = align_and_mask("wtf u r 1d1ot!", "what the fuck | you | are | idiot | !", True, tagger)
    assert r1["alignment_valid"] is True
    assert r1["censored_text"] == "*** u r *****!", r1
    assert len(r1["censored_text"]) == len("wtf u r 1d1ot!")
    assert r1["masked_slot_indices"] == [0, 3]

    # 2. Leetspeak / Symbol Obfuscation
    r2 = align_and_mask("shut up b!tch", "shut | up | bitch", True, tagger)
    assert r2["alignment_valid"] is True
    assert r2["censored_text"] == "shut up *****", r2

    # 3. Spaced Letters (group_letters via Task 1 tokenize)
    r3 = align_and_mask("gg f u c k noob", "gg | fuck | noob", True, tagger)
    assert r3["alignment_valid"] is True
    assert r3["censored_text"] == "gg ******* noob", r3

    # 4. Substring Safety
    r4 = align_and_mask("nice assist in class u a$$", "nice | assist | in | class | you | ass", True, tagger)
    assert r4["alignment_valid"] is True
    assert r4["censored_text"] == "nice assist in class u ***", r4

    # 5. Gatekeeper Bypass (trigger = False must not call predictor)
    def fail_if_called(_words):
        raise AssertionError("predict_word_tags_fn must not run when is_toxic_trigger=False")

    raw5 = "wtf that bkb timing was sick"
    r5 = align_and_mask(raw5, "what the fuck | that | bkb | timing | was | sick", False, fail_if_called)
    assert r5["censored_text"] == raw5
    assert r5["masked_slot_indices"] == []
    assert r5["alignment_valid"] is True


def test_delimiter_mismatch_fallback() -> None:
    """Verify safe fallback to raw_slots when Task 1 delimiter count mismatches."""
    tagger = _stub_tagger({"1d1ot"})
    res = align_and_mask("wtf u r 1d1ot!", "what the fuck | you", True, tagger)
    assert res["alignment_valid"] is False
    assert res["censored_text"] == "wtf u r *****!", res


def test_live_bert_crf_integration() -> list[list[str]]:
    """Run all 5 scenarios through trained BertCRF checkpoint for end-to-end verification."""
    ckpt_path = RUNS_DIR / "baseline" / "bert_crf.pt"
    if not ckpt_path.exists():
        log("SKIP", f"Checkpoint belum ada di {ckpt_path}")
        return []

    pipe = ToxicSpanCensor(ckpt_path=ckpt_path)
    cases = [
        ("1. Token Expansion", "wtf u r 1d1ot!", "what the fuck | you | are | idiot | !", True),
        ("2. Obfuscation", "shut up b!tch", "shut | up | bitch", True),
        ("3. Spaced Letters", "gg f u c k noob", "gg | fuck | noob", True),
        ("4. Substring Safety", "nice assist in class u a$$", "nice | assist | in | class | you | ass", True),
        ("5. Gate Bypass", "wtf that bkb timing was sick", "what the fuck | that | bkb | timing | was | sick", False),
    ]
    rows = [["Skenario", "Trigger", "Censored Output", "Masked Slots", "Align"]]
    for label, raw, norm, trig in cases:
        slots = task1_tokenize(raw)
        res = pipe.censor(raw, norm, is_toxic_trigger=trig, raw_slots=slots)
        assert len(res["censored_text"]) == len(raw)
        assert res["alignment_valid"] is True
        if not trig:
            assert res["censored_text"] == raw
        rows.append([label, str(trig), res["censored_text"], str(res["masked_slot_indices"]), str(res["alignment_valid"])])
    return rows


def main() -> int:
    log("START", "Memulai verifikasi Phase 6 (RC-03 Smart Masking)")
    test_five_rc03_scenarios()
    log("CHECKPOINT 1", "5 skenario unit test RS-03/RC-03 lulus 100%")
    test_delimiter_mismatch_fallback()
    log("CHECKPOINT 2", "Fallback delimiter mismatch lulus")
    live_rows = test_live_bert_crf_integration()
    if live_rows:
        log("CHECKPOINT 3", "Integrasi end-to-end dengan checkpoint BertCRF lulus")
        print_table("Uji End-to-End ToxicSpanCensor (BertCRF)", live_rows)
    log("DONE", "Seluruh verifikasi Phase 6 lulus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
