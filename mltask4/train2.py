"""v2 trainer: ImageNet-pretrained timm encoder + U-Net decoder with a full-resolution branch,
optional input upscaling (helps 1-2 px vessels). Same 5-fold protocol / OOF format as train.py,
so runs can be combined by predict_v4.py (the fixed actual-submission ensemble).
"""
import argparse, json, os, random
import numpy as np
import cv2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import Block, augment, DS, loss_fn, load_img, load_lbl, ids, dice, SEG, OUT

PRE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")


class TimmUNet(nn.Module):
    def __init__(self, enc="resnet34.a1_in1k", pretrained=True, dec=(256, 128, 64, 32, 32)):
        super().__init__()
        kw = dict(pretrained_cfg_overlay=dict(file=f"{PRE}/{enc}.safetensors")) if pretrained else {}
        self.enc = timm.create_model(enc, pretrained=pretrained, features_only=True, in_chans=4, **kw)
        ch = self.enc.feature_info.channels()  # strides 2,4,8,16,32
        self.full = Block(4, 32)  # stride-1 branch for thin vessels
        skips = list(reversed(ch[:-1])) + [32]
        cin = ch[-1]
        self.dec = nn.ModuleList()
        for s, o in zip(skips, dec):
            self.dec.append(Block(cin + s, o)); cin = o
        self.head = nn.Conv2d(cin, 1, 1)

    def forward(self, x):
        f = self.enc((x - 0.5) / 0.25)
        skips = list(reversed(f[:-1])) + [self.full(x)]
        y = f[-1]
        for d, s in zip(self.dec, skips):
            y = F.interpolate(y, size=s.shape[-2:], mode="bilinear", align_corners=False)
            y = d(torch.cat([y, s], 1))
        return self.head(y)


def rescale(x, y, s):
    if s == 1:
        return x, y
    C, H, W = x.shape
    nh, nw = int(round(H * s)), int(round(W * s))
    xs = np.stack([cv2.resize(c, (nw, nh), interpolation=cv2.INTER_CUBIC) for c in x]).clip(0, 1)
    ys = None if y is None else (cv2.resize(y, (nw, nh), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.float32)
    return xs, ys


@torch.no_grad()
def predict2(models, x, scale, tta=True):
    C, H, W = x.shape
    xs, _ = rescale(x, None, scale)
    h, w = xs.shape[1:]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    t = torch.from_numpy(np.pad(xs, ((0, 0), (0, ph), (0, pw)), mode="reflect"))[None].cuda()
    acc = 0; n = 0
    for m in models:
        for f in ([[], [3], [2], [2, 3]] if tta else [[]]):
            ti = torch.flip(t, f) if f else t
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = torch.sigmoid(m(ti).float())
            acc = acc + (torch.flip(o, f) if f else o); n += 1
    p = (acc / n)[0, 0, :h, :w].float().cpu().numpy()
    return cv2.resize(p, (W, H), interpolation=cv2.INTER_AREA) if scale != 1 else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="resnet34.a1_in1k")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--only_fold", type=int, default=-1, help="quick check: train only this fold")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    T = f"{a.tag}_s{a.seed}"
    if a.full:
        T = f"full_{T}"
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True

    tr = ids("train")
    X0, FOV = zip(*[load_img(f"{SEG}/train/image/{i}.jpg") for i in tr])
    Y0 = [load_lbl(f"{SEG}/train/label/{i}.jpg") for i in tr]
    XS, YS = zip(*[rescale(x, y, a.scale) for x, y in zip(X0, Y0)])
    perm = np.random.RandomState(a.seed).permutation(len(tr))  # same split as train.py for equal seed
    folds = [np.array([], dtype=int)] if a.full else np.array_split(perm, a.folds)
    oof = [None] * len(tr)
    for k, va in enumerate(folds):
        if a.only_fold >= 0 and k != a.only_fold:
            continue
        tri = [i for i in range(len(tr)) if i not in va]
        m = TimmUNet(a.enc).cuda().to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
        dl = torch.utils.data.DataLoader(DS([XS[i] for i in tri], [YS[i] for i in tri], a.iters * a.bs, a.crop),
                                         batch_size=a.bs, num_workers=12, drop_last=True, persistent_workers=False,
                                         worker_init_fn=lambda w: (random.seed(a.seed * 1000 + k * 100 + w),
                                                                   np.random.seed(a.seed * 1000 + k * 100 + w)))
        m.train()
        for it, (xb, yb) in enumerate(dl):
            xb = xb.cuda(non_blocking=True).to(memory_format=torch.channels_last); yb = yb.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(xb)
            l = loss_fn(lo.float(), yb)
            opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sch.step()
            if (it + 1) % 1000 == 0:
                print(f"fold {k} it {it+1} loss {l.item():.4f}", flush=True)
        m.eval()
        torch.save(m.state_dict(), f"{OUT}/fold{k}_{T}.pt")
        if a.full:
            print("full model saved", flush=True); return
        for i in va:
            oof[i] = predict2([m], X0[i], a.scale) * FOV[i]
        print(f"fold {k} val dice@0.5 {np.mean([dice(oof[i] > 0.5, Y0[i]) for i in va]):.4f} "
              f"ids {[tr[i] for i in va]}", flush=True)
    if a.only_fold >= 0:
        np.save(f"{OUT}/partial_{T}_f{a.only_fold}.npy", np.stack([oof[i] for i in folds[a.only_fold]]))
        return
    res = {round(float(t), 2): float(np.mean([dice(oof[i] > t, Y0[i]) for i in range(len(tr))]))
           for t in np.arange(0.20, 0.61, 0.02)}
    best = max(res, key=res.get)
    print(f"best th {best} OOF mean dice {res[best]:.4f}", flush=True)
    np.save(f"{OUT}/oof_{T}.npy", np.stack(oof))
    json.dump({"best_th": best, "oof": res, "args": vars(a), "arch": "timm"}, open(f"{OUT}/cv_{T}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
