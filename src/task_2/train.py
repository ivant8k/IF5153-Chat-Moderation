"""Train Task 2: fine-tuning DistilRoBERTa, masked multi-label loss, data fusion Jigsaw + CONDA

Contoh arm eksperimen (dari root repo):
  E1 jigsaw saja     : python src/task_2/train.py --run e1_jigsaw --sources jigsaw
  E2 campuran        : python src/task_2/train.py --run e2_mix --sources jigsaw,conda
  E3 bertahap        : python src/task_2/train.py --run e3_seq --sources conda --init-from e1_jigsaw --lr 1e-5 --select all
"""
import argparse
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from common import build_train, load_set, write_json
from config import BACKBONE, CATS, MAX_LEN, RUNS, SEED
from model import ToxicityModel, autocast_ctx, load_model, pick_device, predict_probs, save_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dev_score(model, tok, device, use_conda, use_jigsaw, limit, max_len):
    """skor seleksi checkpoint: rata-rata AP gate CONDA dev dan AP kategori Jigsaw dev"""
    parts = {}
    if use_conda:
        d = load_set("conda_dev")
        d = d.sample(min(limit, len(d)), random_state=0) if limit else d
        p = predict_probs(model, tok, d["norm"].tolist(), device, max_len=max_len)
        parts["conda_gate_ap"] = float(average_precision_score(d["gate"], p[:, 0]))
    if use_jigsaw:
        d = load_set("jigsaw_dev")
        d = d.sample(min(limit, len(d)), random_state=0) if limit else d
        p = predict_probs(model, tok, d["text"].tolist(), device, max_len=max_len)
        aps = [average_precision_score(d[c], p[:, 1 + k]) for k, c in enumerate(CATS) if d[c].sum() > 0]
        parts["jigsaw_cat_ap"] = float(np.mean(aps))
    return float(np.mean(list(parts.values()))), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sources", default="jigsaw,conda")
    ap.add_argument("--conda-text", default="norm,raw", help="varian teks CONDA untuk latih: norm,raw")
    ap.add_argument("--jigsaw-neg-frac", type=float, default=0.35)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--init-from", default=None, help="run id lain (lanjut dari checkpoint terbaiknya)")
    ap.add_argument("--select", choices=["auto", "all"], default="auto")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true", help="uji cepat di CPU")
    args = ap.parse_args()

    sources = args.sources.split(",")
    conda_text = args.conda_text.split(",")
    if not sources or set(sources) - {"jigsaw", "conda"} or len(sources) != len(set(sources)):
        ap.error("--sources harus berisi jigsaw dan/atau conda tanpa duplikat")
    if not conda_text or set(conda_text) - {"norm", "raw"}:
        ap.error("--conda-text hanya menerima norm dan/atau raw")
    if "conda" in sources and not conda_text:
        ap.error("--conda-text wajib diisi jika CONDA digunakan")
    if not 0 <= args.jigsaw_neg_frac <= 1:
        ap.error("--jigsaw-neg-frac harus berada di antara 0 dan 1")
    if args.epochs < 1 or args.bs < 1 or args.max_len < 1 or args.lr <= 0:
        ap.error("--epochs, --bs, --max-len, dan --lr harus positif")
    out_dir = RUNS / args.run
    if (out_dir / "DONE").exists():
        print(f"{args.run} sudah selesai")
        return
    set_seed(args.seed)
    device = pick_device()
    limit = 300 if args.smoke else None
    if args.smoke:
        args.epochs, args.bs, args.max_len = 1, 8, 32

    texts, Y, M = build_train(sources, conda_text, args.jigsaw_neg_frac, args.seed, limit)
    if not texts:
        ap.error("Tidak ada contoh training; jalankan prepare.py dan periksa sumber data")
    print(f"{args.run} | device {device} | train {len(texts)} | positif gate {Y[:, 0].mean():.3f}", flush=True)

    # pos_weight kategori dari data latih (akar rasio neg/pos, maks 10); gate tanpa bobot
    pw = np.ones(Y.shape[1], dtype=np.float32)
    for k in range(1, Y.shape[1]):
        valid = M[:, k] > 0
        pos, neg = Y[valid, k].sum(), valid.sum() - Y[valid, k].sum()
        pw[k] = min(np.sqrt(neg / max(pos, 1)), 10.0)
    pos_weight = torch.tensor(pw, device=device)

    if args.init_from:
        tok, model = load_model(RUNS / args.init_from / "checkpoints" / "best", device)
    else:
        tok = AutoTokenizer.from_pretrained(BACKBONE)
        model = ToxicityModel().to(device)
    model.train()

    def collate(idx):
        enc = tok([texts[i] for i in idx], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"], torch.tensor(Y[idx]), torch.tensor(M[idx])

    loader = DataLoader(list(range(len(texts))), batch_size=args.bs, shuffle=True, collate_fn=collate)
    steps = len(loader) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    use_fp16 = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    use_conda = "conda" in sources or args.select == "all"
    use_jigsaw = "jigsaw" in sources or args.select == "all"
    best, log, t0 = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for step, (ids, att, y, m) in enumerate(loader, 1):
            ids, att, y, m = ids.to(device), att.to(device), y.to(device), m.to(device)
            with autocast_ctx(device):
                logits = model(ids, att)
            loss = (bce(logits.float(), y) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if step % 200 == 0:
                print(f"  ep{epoch} step {step}/{len(loader)} loss {total / step:.4f} ({time.time() - t0:.0f}s)", flush=True)
        score, parts = dev_score(model, tok, device, use_conda, use_jigsaw, 200 if args.smoke else None, args.max_len)
        rec = {"epoch": epoch, "train_loss": total / len(loader), "score": score, **parts}
        log.append(rec)
        print(rec, flush=True)
        if score > best:
            best = score
            save_model(model, tok, out_dir / "checkpoints" / "best")

    write_json(out_dir / "config.json", {**vars(args), "kind": "torch", "n_train": len(texts), "pos_weight": pw.tolist(),
                                        "best_score": best, "log": log, "seconds": time.time() - t0})
    (out_dir / "DONE").write_text("ok\n")
    print(f"selesai {args.run} | best {best:.4f}")


if __name__ == "__main__":
    main()
