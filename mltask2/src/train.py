"""Step 3/4: train one fold of one architecture; writes OOF + test probabilities.

Usage: python src/train.py --arch convnext_tiny --fold 0 (full recipe in README).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    ART,
    CLASSES,
    ModelEma,
    PixelDataset,
    augment,
    build_model,
    load_data,
    mixup_cutmix,
    soft_ce,
    to_model_input,
)

# Historical contributing runs used the plain CNN path, without an STN wrapper.
AUG = dict(rot=12.0, trans=0.10, scale=0.12, shear=6.0, bright=0.25,
           contrast=0.25, erase_p=0.35, erase_max=0.20)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="convnext_tiny")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--img-size", type=int, default=112)
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--head-lr-mult", type=float, default=10.0)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--mix-prob", type=float, default=0.5)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--drop-rate", type=float, default=0.2)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20261011)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tag", default="")
    p.add_argument("--max-train-batches", type=int, default=0)
    # Fixed to the original exact-duplicate grouped fold.npy.
    return p.parse_args()


def head_param_names(model):
    names = []
    for n, _ in model.named_parameters():
        if n.startswith(("fc.", "classifier.", "head.")):
            names.append(n)
    return set(names)


def make_model(a, drop_rate, drop_path, device):
    """Single construction path so training and evaluation always agree."""
    model = build_model(a.arch, len(CLASSES), drop_rate, drop_path)
    return model.to(device).to(memory_format=torch.channels_last)


@torch.no_grad()
def predict(model, pixels, indices, size, device, batch=256, tta=True):
    """Plain backbones consume the normalised three-channel tensor."""
    model.eval()
    out = np.zeros((len(indices), len(CLASSES)), dtype=np.float32)
    for s in range(0, len(indices), batch):
        sel = indices[s: s + batch]
        x = torch.from_numpy(pixels[sel]).to(device).float().div_(255).unsqueeze(1)
        prep = lambda t: to_model_input(t, size)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = F.softmax(model(prep(x)).float(), dim=1)
            if tta:
                p = p + F.softmax(model(prep(x.flip(-1))).float(), dim=1)
        out[s: s + len(sel)] = (p / (2 if tta else 1)).float().cpu().numpy()
    return out


def main():
    a = parse()
    device = "cuda"
    torch.manual_seed(a.seed + a.fold)
    np.random.seed(a.seed + a.fold)
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device=device)
    gen.manual_seed(a.seed * 1000 + a.fold)

    tr_px, te_px, y, fold, _ = load_data()
    tr_idx = np.where(fold != a.fold)[0]
    va_idx = np.where(fold == a.fold)[0]

    model = make_model(a, a.drop_rate, a.drop_path, device)
    heads = head_param_names(model)
    decay, no_decay, head_p = [], [], []
    for n, p in model.named_parameters():
        if n in heads:
            head_p.append(p)
        elif p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": a.weight_decay, "lr": a.lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": a.lr},
            {"params": head_p, "weight_decay": a.weight_decay,
             "lr": a.lr * a.head_lr_mult},
        ]
    )
    base_lrs = [g["lr"] for g in opt.param_groups]

    ds = PixelDataset(tr_px, y, tr_idx)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)
    steps_per_epoch = len(dl) if a.max_train_batches == 0 else min(len(dl), a.max_train_batches)
    total_steps = steps_per_epoch * a.epochs
    warmup = max(1, int(total_steps * a.warmup_frac))

    ema = ModelEma(model, a.ema_decay)
    tag = a.tag or a.arch
    log_path = os.path.join(ROOT_LOGS, f"{tag}_fold{a.fold}.jsonl")
    best = {"acc": -1.0, "epoch": -1}
    step = 0
    t0 = time.time()

    for epoch in range(a.epochs):
        model.train()
        tr_loss, tr_correct, tr_seen = 0.0, 0, 0
        for bi, (xb, yb) in enumerate(dl):
            if a.max_train_batches and bi >= a.max_train_batches:
                break
            lr_scale = (
                (step + 1) / warmup
                if step < warmup
                else 0.5 * (1 + math.cos(math.pi * (step - warmup) /
                                         max(1, total_steps - warmup)))
            )
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * lr_scale
            xb = xb.to(device, non_blocking=True).float().div_(255).unsqueeze(1)
            yb = yb.to(device, non_blocking=True)
            xb = augment(xb, gen, AUG)
            xb, target = mixup_cutmix(xb, yb, gen, a.mixup, a.cutmix, a.mix_prob,
                                      len(CLASSES))
            xi = to_model_input(xb, a.img_size).to(
                memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xi)
                loss = soft_ce(logits.float(), target, a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)
            step += 1
            tr_loss += loss.item() * xb.shape[0]
            tr_correct += (logits.float().argmax(1) == yb).sum().item()
            tr_seen += xb.shape[0]

        # validation with EMA weights
        eval_model = make_model(a, 0.0, 0.0, device)
        ema.copy_to(eval_model)
        va_prob = predict(eval_model, tr_px, va_idx, a.img_size, device, tta=False)
        va_acc = float((va_prob.argmax(1) == y[va_idx]).mean())
        tr_acc = tr_correct / max(1, tr_seen)
        rec = {
            "epoch": epoch,
            "train_loss": round(tr_loss / max(1, tr_seen), 4),
            "train_acc_on_augmented_mixed": round(tr_acc, 4),
            "val_acc_ema": round(va_acc, 4),
            "lr": round(opt.param_groups[0]["lr"], 7),
            "elapsed_s": round(time.time() - t0, 1),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if va_acc > best["acc"]:
            best = {"acc": va_acc, "epoch": epoch}
            torch.save({k: v.half() for k, v in ema.shadow.items()},
                       os.path.join(ART, f"{tag}_fold{a.fold}.pt"))
        del eval_model
        torch.cuda.empty_cache()

    # final predictions from the best EMA checkpoint
    eval_model = make_model(a, 0.0, 0.0, device)
    sd = torch.load(os.path.join(ART, f"{tag}_fold{a.fold}.pt"), map_location="cpu")
    eval_model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)

    va_plain = predict(eval_model, tr_px, va_idx, a.img_size, device, tta=False)
    va_tta = predict(eval_model, tr_px, va_idx, a.img_size, device, tta=True)
    te_plain = predict(eval_model, te_px, np.arange(len(te_px)), a.img_size, device,
                       tta=False)
    te_tta = predict(eval_model, te_px, np.arange(len(te_px)), a.img_size, device,
                     tta=True)
    # train-partition accuracy without augmentation: overfitting diagnostic
    sub = np.random.RandomState(0).choice(tr_idx, size=min(6000, len(tr_idx)),
                                          replace=False)
    tr_clean = predict(eval_model, tr_px, sub, a.img_size, device, tta=False)
    tr_clean_acc = float((tr_clean.argmax(1) == y[sub]).mean())

    np.savez_compressed(
        os.path.join(ART, f"{tag}_fold{a.fold}_preds.npz"),
        va_idx=va_idx, va_plain=va_plain, va_tta=va_tta,
        te_plain=te_plain, te_tta=te_tta,
    )
    summary = {
        "arch": a.arch, "tag": tag, "fold": a.fold,
        "config": vars(a), "aug": AUG, "split": "fold.npy",
        "best_epoch": best["epoch"],
        "val_acc_plain": round(float((va_plain.argmax(1) == y[va_idx]).mean()), 4),
        "val_acc_tta_flip": round(float((va_tta.argmax(1) == y[va_idx]).mean()), 4),
        "train_partition_clean_acc": round(tr_clean_acc, 4),
        "generalisation_gap": round(
            tr_clean_acc - float((va_plain.argmax(1) == y[va_idx]).mean()), 4),
        "minutes": round((time.time() - t0) / 60, 2),
    }
    with open(os.path.join(ART, f"{tag}_fold{a.fold}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


ROOT_LOGS = os.path.join(os.path.dirname(ART), "logs")
os.makedirs(ROOT_LOGS, exist_ok=True)

if __name__ == "__main__":
    main()
