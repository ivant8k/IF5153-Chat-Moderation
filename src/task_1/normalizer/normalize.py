"""Normalize: src ber-delimiter -> slot normal (inferensi run final)"""
import torch

from align import postprocess
from common import read_json, to_input
from config import BACKBONES, DELIM
from model import generate, load_checkpoint, setup_device
from runs import run_dir


class Normalizer:
    def __init__(self, run_id, device=None, beams=1):
        spec = read_json(run_dir(run_id) / "config.json")
        self.fmt = spec["fmt"]
        self.beams = beams
        self.batch = max(1, BACKBONES[spec["backbone"]]["gen_batch"] // beams)
        self.device = torch.device(device) if device else setup_device()
        self.tok, self.model = load_checkpoint(run_dir(run_id) / "checkpoints" / "best", self.device)

    def normalize(self, srcs):
        """list src -> list (slot normal, valid)"""
        todo = [i for i, s in enumerate(srcs) if s]
        outs = generate(self.tok, self.model, [to_input(srcs[i], self.fmt) for i in todo], self.batch, self.beams)
        result = [([], True)] * len(srcs)
        for i, out in zip(todo, outs):
            result[i] = postprocess(srcs[i], out, self.fmt)
        return result

    def normalize_text(self, src):
        """satu src -> teks ber-delimiter"""
        slots, _ = self.normalize([src])[0]
        return DELIM.join(slots)
