"""Model: load backbone/checkpoint, generate, info device"""
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import BACKBONES, MAX_LEN, PRETRAINED


def setup_device():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info():
    if not torch.cuda.is_available():
        return {"device": "cpu", "precision": "fp32"}
    name = torch.cuda.get_device_name(0)
    major, _ = torch.cuda.get_device_capability(0)
    return {"device": name, "precision": "tf32" if major >= 8 else "fp32",
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
            "torch": torch.__version__}


def load_backbone(name, device):
    """nama backbone -> tokenizer, model (lokal jika ada)"""
    local = PRETRAINED / name
    if local.exists():
        tok = AutoTokenizer.from_pretrained(local)
        model = AutoModelForSeq2SeqLM.from_pretrained(local)
    else:
        info = BACKBONES[name]
        tok = AutoTokenizer.from_pretrained(info["hf"], revision=info["rev"])
        model = AutoModelForSeq2SeqLM.from_pretrained(info["hf"], revision=info["rev"])
    return tok, model.to(device)


def load_checkpoint(path, device):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return tok, model.to(device)


def decode(tok, ids):
    # <unk> dipertahankan, pad dan eos dibuang
    text = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for special in (tok.pad_token, tok.eos_token):
        text = text.replace(special, "")
    return " ".join(text.split())


@torch.no_grad()
def generate(tok, model, texts, batch_size, beams=1):
    """list teks input -> list teks output"""
    model.eval()
    device = next(model.parameters()).device
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outs = [None] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt").to(device)
        ids = model.generate(**enc, max_new_tokens=MAX_LEN, num_beams=beams, do_sample=False)
        for i, row in zip(idx, ids):
            outs[i] = decode(tok, row)
    return outs


def latency(tok, model, texts, beams=1):
    """list teks -> ms per baris (batch 1)"""
    times = []
    for text in texts:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        generate(tok, model, [text], 1, beams)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return times
