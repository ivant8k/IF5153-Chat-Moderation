"""Model Task 2: DistilRoBERTa + head 7 output (gate + 6 kategori Jigsaw), predictor"""
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from common import read_json, write_json
from config import BACKBONE, CATS, LABELS, MAX_LEN


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ToxicityModel(nn.Module):
    """encoder -> mean pooling -> dropout -> linear(7)"""

    def __init__(self, backbone=BACKBONE, config_dir=None, dropout=0.1):
        super().__init__()
        if config_dir is None:
            self.enc = AutoModel.from_pretrained(backbone)
        else:   # muat dari checkpoint sendiri, tanpa unduh
            self.enc = AutoModel.from_config(AutoConfig.from_pretrained(config_dir))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.enc.config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))


def save_model(model, tok, path, backbone=BACKBONE):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    model.enc.config.save_pretrained(path)
    tok.save_pretrained(path)
    write_json(path / "meta.json", {"backbone": backbone, "labels": LABELS})


def load_model(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    model = ToxicityModel(config_dir=path)
    model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
    return tok, model.to(device).eval()


def autocast_ctx(device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def predict_probs(model, tok, texts, device, bs=128, max_len=MAX_LEN):
    """texts -> (n, 7) probabilitas sigmoid, urutan asli"""
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        with autocast_ctx(device):
            logits = model(**enc)
        out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


class TorchPredictor:
    """checkpoint hasil train.py -> .probs(texts)"""

    def __init__(self, path, device=None):
        self.device = device or pick_device()
        self.tok, self.model = load_model(path, self.device)

    def probs(self, texts):
        return predict_probs(self.model, self.tok, texts, self.device)


class ZeroShotPredictor:
    """model toksisitas siap pakai (tanpa fine-tuning) -> format 7 output.
    6 label Jigsaw (unitary/toxic-bert): gate = max enam label. Biner: gate = P(toxic), kategori = 0."""

    def __init__(self, hf_id, device=None):
        self.device = device or pick_device()
        self.tok = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_id).to(self.device).eval()
        names = {i: n.lower().replace(" ", "_") for i, n in self.model.config.id2label.items()}
        self.cat_idx = {c: i for i, n in names.items() for c in CATS if n == c}
        self.binary = len(self.cat_idx) < len(CATS)
        self.toxic_idx = next((i for i, n in names.items() if "toxic" in n and "non" not in n and "not" not in n),
                              len(names) - 1)

    @torch.no_grad()
    def probs(self, texts, bs=128):
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = self.tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True,
                           return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.float()
            if self.binary:
                out[idx, 0] = torch.softmax(logits, -1)[:, self.toxic_idx].cpu().numpy()
            else:
                p = torch.sigmoid(logits).cpu().numpy()
                for c, k in self.cat_idx.items():
                    out[idx, 1 + CATS.index(c)] = p[:, k]
                out[idx, 0] = out[idx, 1:].max(axis=1)
        return out
