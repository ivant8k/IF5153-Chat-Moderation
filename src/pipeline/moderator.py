"""Moderator: chat mentah -> normalisasi (T1) -> klasifikasi (T2) -> span + masking (T3) -> chat tersensor"""
import time
import warnings
from unittest.mock import patch

import torch
from huggingface_hub.utils import disable_progress_bars
from transformers import AutoConfig, AutoModel
from transformers.utils import logging as hf_logging

from loader import task_scope
from settings import DELIM, MASK, T1_DIR, T1_PREPARE_DIR, T1_RUN, T2_DIR, T2_MODE, T2_RUN, T3_BACKBONE, T3_CKPT, T3_DIR


# Quiet Loading
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()
disable_progress_bars()
warnings.filterwarnings("ignore")


def encoder_from_config(name, config=None, **kwargs):
    return AutoModel.from_config(config or AutoConfig.from_pretrained(name))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


class Moderator:
    def __init__(self, device=None, mode=T2_MODE):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with task_scope(T1_PREPARE_DIR):
            import common
            self.tokenize, self.clean_slot = common.tokenize, common.clean_slot
        with task_scope(T1_DIR):
            import normalize
            self.t1 = normalize.Normalizer(T1_RUN, self.device)
        with task_scope(T2_DIR):
            import predict
            self.t2 = predict.ToxicityClassifier(T2_RUN, mode, self.device)
        # encoder dibuat dari config, bobot pretrained tidak perlu karena ditimpa checkpoint
        with task_scope(T3_DIR), patch.object(AutoModel, "from_pretrained", encoder_from_config):
            import predict
            self.t3 = predict.ToxicSpanCensor(T3_CKPT, str(T3_BACKBONE), self.device)

    def set_mode(self, mode):
        if mode not in ("balanced", "strict"):
            raise ValueError("mode harus 'balanced' atau 'strict'")
        self.t2.mode = mode

    def prepare(self, lines):
        """list baris chat -> teks raw gabungan, slot raw (offset di teks gabungan), src per baris"""
        raw_slots, srcs, offset = [], [], 0
        for line in lines:
            toks = self.tokenize(line)
            raw_slots += [(t, s + offset, e + offset) for t, s, e in toks]
            srcs.append(DELIM.join(self.clean_slot(t) for t, _, _ in toks))
            offset += len(line) + 1
        return " ".join(lines), raw_slots, srcs

    def normalize(self, rows, preps=None):
        """list baris -> slot normal per baris, valid per baris"""
        preps = preps or [self.prepare(lines) for lines in rows]
        flat = [src for _, _, srcs in preps for src in srcs]
        outs = iter(self.t1.normalize(flat))
        norm_slots, valid = [], []
        for _, _, srcs in preps:
            parts = [next(outs) for _ in srcs]
            norm_slots.append([s for slots, _ in parts for s in slots])
            valid.append(all(v for _, v in parts))
        return norm_slots, valid

    def run(self, rows, norm="t1", valid=None, gate=True):
        """list baris (list chat) -> list hasil, detik per tahap"""
        # norm: "t1", "none" (slot raw lowercase), atau list slot normal per baris (cache T1 / oracle)
        # gate False: T3 jalan tanpa melihat sinyal T2
        times = {"t1": 0.0, "t2": 0.0, "t3": 0.0}
        preps = [self.prepare(lines) for lines in rows]

        # Normalization
        sync(self.device)
        start = time.perf_counter()
        if norm == "t1":
            norm_slots, valid = self.normalize(rows, preps)
        elif norm == "none":
            norm_slots = [[t.lower() for t, _, _ in raw_slots] for _, raw_slots, _ in preps]
            valid = [True] * len(rows)
        else:
            norm_slots, valid = norm, valid or [True] * len(rows)
        sync(self.device)
        times["t1"] = time.perf_counter() - start
        delim_norm = [DELIM.join(slots) for slots in norm_slots]

        # Classification
        start = time.perf_counter()
        cls = self.t2.classify(delim_norm) if rows else []
        sync(self.device)
        times["t2"] = time.perf_counter() - start

        # Span Extraction and Masking
        results = []
        start = time.perf_counter()
        for (raw, raw_slots, srcs), dn, ok, c in zip(preps, delim_norm, valid, cls):
            trigger = c["trigger"] if gate else True
            out = self.t3.censor(raw, dn, trigger, raw_slots=raw_slots, mask_char=MASK)
            results.append({
                "raw": raw, "src": DELIM.join(srcs), "norm": dn, "t1_valid": ok,
                "raw_slots": raw_slots, "p_toxic": c["p_toxic"], "threshold": c["threshold"], "trigger": c["trigger"],
                "warnings": c["warnings"], "message": c["message"],
                "tags": out["predicted_bio_tags"], "masked_slots": out["masked_slot_indices"],
                "align_valid": out["alignment_valid"], "censored": out["censored_text"],
            })
        sync(self.device)
        times["t3"] = time.perf_counter() - start
        return results, times

    def moderate(self, text):
        """satu pesan chat -> hasil + latensi ms per tahap"""
        results, times = self.run([[text]])
        out = results[0]
        out["ms"] = {k: round(v * 1000, 1) for k, v in times.items()}
        return out
