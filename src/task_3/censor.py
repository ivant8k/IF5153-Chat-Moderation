"""Task 3 RC-03 delimiter-based 1-to-1 alignment and smart masking."""
import importlib.util
from pathlib import Path
import sys
from typing import Callable, Sequence

if __package__:
    from .config import ROOT
else:
    TASK_DIR = str(Path(__file__).resolve().parent)
    if TASK_DIR not in sys.path:
        sys.path.insert(0, TASK_DIR)
    from config import ROOT


def _load_task1_tokenize() -> Callable[[str], list[tuple[str, int, int]]]:
    """Load Task 1 slot tokenizer without polluting sys.modules['config']."""
    prepare_dir = ROOT / "src" / "task_1" / "prepare"
    prev_config = sys.modules.get("config")
    try:
        cfg_spec = importlib.util.spec_from_file_location("config", prepare_dir / "config.py")
        cfg_mod = importlib.util.module_from_spec(cfg_spec)
        assert cfg_spec.loader is not None
        cfg_spec.loader.exec_module(cfg_mod)
        sys.modules["config"] = cfg_mod

        common_spec = importlib.util.spec_from_file_location("task1_prepare_common", prepare_dir / "common.py")
        common_mod = importlib.util.module_from_spec(common_spec)
        assert common_spec.loader is not None
        common_spec.loader.exec_module(common_mod)
        return common_mod.tokenize
    finally:
        if prev_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = prev_config


task1_tokenize = _load_task1_tokenize()


def align_and_mask(
    raw_text: str,
    delim_norm_text: str,
    is_toxic_trigger: bool,
    predict_word_tags_fn: Callable[[list[str]], Sequence[str]],
    raw_slots: list[tuple[str, int, int]] | None = None,
    mask_char: str = "*",
) -> dict:
    """Map normalized-word BIO predictions back to raw_text character spans via 1-to-1 slots."""
    if not is_toxic_trigger:
        return {
            "censored_text": raw_text,
            "masked_slot_indices": [],
            "predicted_bio_tags": [],
            "alignment_valid": True,
        }

    if raw_slots is None:
        raw_slots = task1_tokenize(raw_text)
    if not raw_slots:
        return {
            "censored_text": raw_text,
            "masked_slot_indices": [],
            "predicted_bio_tags": [],
            "alignment_valid": True,
        }

    norm_slots = [s.strip() for s in delim_norm_text.split("|")]
    alignment_valid = len(norm_slots) == len(raw_slots)
    if not alignment_valid:
        norm_slots = [tok for tok, _, _ in raw_slots]

    words: list[str] = []
    word_slot_owners: list[int] = []
    for slot_idx, slot_text in enumerate(norm_slots):
        slot_words = slot_text.split() or [raw_slots[slot_idx][0]]
        for word in slot_words:
            words.append(word)
            word_slot_owners.append(slot_idx)

    pred_bio_tags = list(predict_word_tags_fn(words))
    if len(pred_bio_tags) != len(words):
        raise ValueError(f"Tag count mismatch: expected {len(words)}, got {len(pred_bio_tags)}")

    toxic_slot_indices = sorted({
        word_slot_owners[word_idx]
        for word_idx, tag in enumerate(pred_bio_tags)
        if tag in ("B-TOXIC", "I-TOXIC")
    })

    chars = list(raw_text)
    for slot_idx in toxic_slot_indices:
        _, start_char, end_char = raw_slots[slot_idx]
        for pos in range(start_char, end_char):
            chars[pos] = mask_char

    return {
        "censored_text": "".join(chars),
        "masked_slot_indices": toxic_slot_indices,
        "predicted_bio_tags": list(zip(words, pred_bio_tags)),
        "alignment_valid": alignment_valid,
    }
