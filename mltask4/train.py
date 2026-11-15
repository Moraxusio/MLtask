"""DRIVE vessel segmentation: U-Net, 5-fold CV, OOF threshold tuning, test ensemble + TTA.
Writes checkpoints and OOF probabilities; fixed-v4 export is handled by predict_v4.py.
"""
import argparse, glob, json, os, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
SEG = os.path.join(ROOT, "segmentation")
OUT = os.path.join(ROOT, "work")


def load_img(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fov = (rgb[..., 0] > 25).astype(np.uint8)
    fov = cv2.morphologyEx(fov, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    g = rgb[..., 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    x = np.stack([rgb[..., 0], g, rgb[..., 2], clahe], 0).astype(np.float32) / 255.0
    return x, fov


def load_lbl(path):
    return (cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)


def ids(split):
    return sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{SEG}/{split}/image/*.jpg")], key=int)


# ---------------- model ----------------
class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                               nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

    def forward(self, x):
        return self.c(x)


class UNet(nn.Module):
    def __init__(self, cin=4, base=32, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([Block(cin, ch[0])] + [Block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, 2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([Block(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = self.drop(skips.pop())
        for u, d in zip(self.up, self.dec):
            x = d(torch.cat([u(x), skips.pop()], 1))
        return self.head(x)


# ---------------- data ----------------
def augment(x, y, crop):
    C, H, W = x.shape
    # random rotation/scale
    ang = random.uniform(-180, 180); sc = random.uniform(0.8, 1.2)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
    xs = np.stack([cv2.warpAffine(c, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for c in x])
    ys = cv2.warpAffine(y, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.5:
        xs, ys = xs[:, :, ::-1], ys[:, ::-1]
    if random.random() < 0.5:
        xs, ys = xs[:, ::-1], ys[::-1]
    # intensity jitter
    xs = xs * random.uniform(0.8, 1.2) + random.uniform(-0.1, 0.1)
    gamma = random.uniform(0.7, 1.4)
    xs = np.clip(xs, 0, 1) ** gamma
    i = random.randint(0, H - crop); j = random.randint(0, W - crop)
    return np.ascontiguousarray(xs[:, i:i + crop, j:j + crop]), np.ascontiguousarray(ys[i:i + crop, j:j + crop])


class DS(torch.utils.data.Dataset):
    def __init__(self, X, Y, n, crop):
        self.X, self.Y, self.n, self.crop = X, Y, n, crop

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        i = random.randrange(len(self.X))
        x, y = augment(self.X[i], self.Y[i], self.crop)
        return torch.from_numpy(x), torch.from_numpy(y)[None]


def loss_fn(logit, y):
    bce = F.binary_cross_entropy_with_logits(logit, y)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


# ---------------- inference ----------------
PAD_H, PAD_W = 592, 576  # multiples of 16


@torch.no_grad()
def predict(models, x, tta=True):
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, PAD_H - H), (0, PAD_W - W)), mode="reflect")
    t = torch.from_numpy(xp)[None].cuda()
    acc = torch.zeros(1, 1, PAD_H, PAD_W, device="cuda"); n = 0
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    for m in models:
        for f in flips:
            ti = torch.flip(t, f) if f else t
            o = torch.sigmoid(m(ti))
            acc += torch.flip(o, f) if f else o; n += 1
    return (acc / n)[0, 0, :H, :W].cpu().numpy()


def dice(p, g):
    return 2 * (p * g).sum() / (p.sum() + g.sum() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--full", action="store_true", help="train one model on all 20 images, no validation")
    a = ap.parse_args()
    T = f"{a.tag}_s{a.seed}" if a.tag else f"s{a.seed}"
    os.makedirs(OUT, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    tr = ids("train")
    X, FOV = zip(*[load_img(f"{SEG}/train/image/{i}.jpg") for i in tr])
    Y = [load_lbl(f"{SEG}/train/label/{i}.jpg") for i in tr]
    perm = np.random.RandomState(a.seed).permutation(len(tr))
    folds = np.array_split(perm, a.folds)
    if a.full:
        folds = [np.array([], dtype=int)]
        T = f"full_{T}"
    oof = [None] * len(tr)
    models = []
    for k, va in enumerate(folds):
        tri = [i for i in range(len(tr)) if i not in va]
        m = UNet(base=a.base).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
        dl = torch.utils.data.DataLoader(DS([X[i] for i in tri], [Y[i] for i in tri], a.iters * a.bs, a.crop),
                                         batch_size=a.bs, num_workers=12, drop_last=True,
                                         worker_init_fn=lambda w: (random.seed(a.seed * 1000 + k * 100 + w),
                                                                   np.random.seed(a.seed * 1000 + k * 100 + w)))
        m.train()
        for it, (xb, yb) in enumerate(dl):
            xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(xb)
            l = loss_fn(lo.float(), yb)
            opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sch.step()
            if (it + 1) % 1000 == 0:
                print(f"fold {k} it {it+1} loss {l.item():.4f}", flush=True)
        m.eval()
        torch.save(m.state_dict(), f"{OUT}/fold{k}_{T}.pt")
        if a.full:
            print("full model saved", flush=True)
            return
        ds_ = []
        for i in va:
            oof[i] = predict([m], X[i]) * FOV[i]
            ds_.append(dice(oof[i] > 0.5, Y[i]))
        print(f"fold {k} val dice@0.5 {np.mean(ds_):.4f} ids {[tr[i] for i in va]}", flush=True)
        models.append(m)

    # threshold tuning on OOF (mean per-image Dice, vessel=positive)
    res = {}
    for th in np.arange(0.30, 0.71, 0.02):
        res[round(float(th), 2)] = float(np.mean([dice(oof[i] > th, Y[i]) for i in range(len(tr))]))
    best = max(res, key=res.get)
    print("OOF dice by threshold:", json.dumps(res))
    print(f"best th {best} OOF mean dice {res[best]:.4f}")
    np.save(f"{OUT}/oof_{T}.npy", np.stack(oof))
    json.dump({"best_th": best, "oof": res, "args": vars(a)}, open(f"{OUT}/cv_{T}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
