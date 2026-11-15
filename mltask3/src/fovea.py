"""Stage-1 fovea localisation (ConvNeXt-Small heatmap + soft-argmax).

Whole half-cache images are letterboxed to 1024x704 or 512x352.
Both CV and full-data training are retained for the actual four-group submission.
The current reg head is retained for initialization/RNG compatibility, not loss.
Historical CV checkpoints lack that head; see README.md for reproduction limits.

All geometric augmentation is a single 2x3 affine applied with grid_sample on the
GPU, so the target point is transformed exactly by the same matrix.
Coordinates: pixel-centre convention (pixel i has centre i). Full-res x_f maps to
half-res x_h = (x_f + 0.5) / 2 - 0.5.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(os.environ.get("MLTASK3_DATA_DIR", f"{ROOT}/data/detection"))
CACHE = f"{ROOT}/out/half_cache.pt"
WEIGHTS = os.environ.get("MLTASK3_WEIGHTS_DIR", f"{ROOT}/weights")
DEV = "cuda"

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ----------------------------------------------------------------------------- data
def load_cache():
    if os.path.exists(CACHE):
        return torch.load(CACHE)
    out = {}
    for f in sorted(glob.glob(f"{DATA}/train/*.jpg") + glob.glob(f"{DATA}/test/*.jpg")):
        iid = int(os.path.basename(f)[:4])
        im = torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).permute(2, 0, 1)
        h, w = im.shape[1:]
        half = F.interpolate(im[None].float(), size=(h // 2, w // 2), mode="bilinear",
                             antialias=True, align_corners=False)[0]
        out[iid] = {"img": half.round().clamp(0, 255).to(torch.uint8), "full_size": (w, h)}
    torch.save(out, CACHE)
    return out


def full_to_half(p):
    return (np.asarray(p, dtype=np.float64) + 0.5) / 2 - 0.5


def half_to_full(p):
    return (np.asarray(p, dtype=np.float64) + 0.5) * 2 - 0.5


def affine(center_src, center_dst, scale, rot_deg, flip):
    """Matrix mapping source(half-res) pixel -> output pixel."""
    a = math.radians(rot_deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    R = np.array([[c, -s], [s, c]])
    if flip:
        R = R @ np.array([[-1, 0], [0, 1]])
    t = np.asarray(center_dst) - R @ np.asarray(center_src)
    return np.hstack([R, t[:, None]])


def warp(img_u8, M, out_wh):
    """img_u8 (3,H,W) half-cache uint8 on GPU, M 2x3 src px->dst; returns float 0..1."""
    Wo, Ho = out_wh
    H, W = img_u8.shape[1:]
    A = np.vstack([M, [0, 0, 1]])
    Ai = torch.tensor(np.linalg.inv(A)[:2], dtype=torch.float32, device=DEV)
    v, u = torch.meshgrid(torch.arange(Ho, device=DEV, dtype=torch.float32),
                          torch.arange(Wo, device=DEV, dtype=torch.float32), indexing="ij")
    pts = torch.stack([u, v, torch.ones_like(u)], -1) @ Ai.T  # (Ho,Wo,2) src px
    gx = (2 * pts[..., 0] + 1) / W - 1
    gy = (2 * pts[..., 1] + 1) / H - 1
    grid = torch.stack([gx, gy], -1)[None]
    x = F.grid_sample(img_u8[None].float() / 255, grid, mode="bilinear",
                      padding_mode="zeros", align_corners=False)
    return x[0]


def color_aug(x, rng):
    # x (3,H,W) 0..1 ; brightness/contrast/gamma/per-channel gain
    g = torch.tensor(rng.uniform(0.85, 1.15, 3), dtype=torch.float32, device=DEV).view(3, 1, 1)
    x = x * g * rng.uniform(0.75, 1.25)
    m = x.mean()
    x = (x - m) * rng.uniform(0.75, 1.3) + m
    x = x.clamp(0, 1) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.3:
        x = x + torch.randn_like(x) * rng.uniform(0, 0.03)
    return x.clamp(0, 1)


def normalize(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


# ----------------------------------------------------------------------------- model
class HeatNet(nn.Module):
    WFILES = {"convnext_small": f"{WEIGHTS}/convnext_small.fb_in22k_ft_in1k.safetensors"}

    def __init__(self, arch="convnext_small"):
        super().__init__()
        import safetensors.torch as st
        import timm
        sd = st.load_file(self.WFILES[arch])
        self.bb = timm.create_model(arch, pretrained=False, num_classes=0, drop_path_rate=0.1)
        sd = {k: v for k, v in sd.items() if not k.startswith("head.fc")}
        r = self.bb.load_state_dict(sd, strict=False)
        bad = [k for k in r.missing_keys if not k.startswith("head")]
        assert not bad and not r.unexpected_keys, (bad, r.unexpected_keys)
        chs = [s.blocks[-1].mlp.fc2.out_features for s in self.bb.stages]
        C = 128
        self.lat = nn.ModuleList([nn.Conv2d(c, C, 1) for c in chs])
        # Keep this head and its initialization order: removing it changes the RNG
        # state used by self.head. Historical CV models had no reg; not reconstructed here.
        self.reg = nn.Sequential(nn.Linear(chs[3], 256), nn.GELU(), nn.Linear(256, 2))
        nn.init.zeros_(self.reg[-1].weight); nn.init.zeros_(self.reg[-1].bias)
        self.head = nn.Sequential(nn.Conv2d(C, C, 3, padding=1), nn.GroupNorm(16, C), nn.GELU(),
                                  nn.Conv2d(C, C, 3, padding=1), nn.GroupNorm(16, C), nn.GELU(),
                                  nn.Conv2d(C, 1, 1))

    def forward(self, x):
        x = self.bb.stem(x)
        feats = []
        for s in self.bb.stages:
            x = s(x)
            feats.append(x)
        y = self.lat[3](feats[3])
        for i in (2, 1, 0):
            y = F.interpolate(y, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            y = y + self.lat[i](feats[i])
        self._reg = self.reg(feats[3].mean((2, 3)).float())  # retained computation; no auxiliary loss
        return self.head(y)[:, 0]  # (B,h,w) logits at stride 4


STRIDE = 4


def hm_to_px(c):
    return c * STRIDE + (STRIDE - 1) / 2


def soft_argmax(logits):
    B, h, w = logits.shape
    p = logits.float().flatten(1).softmax(1).view(B, h, w)
    xs = hm_to_px(torch.arange(w, device=logits.device, dtype=torch.float32))
    ys = hm_to_px(torch.arange(h, device=logits.device, dtype=torch.float32))
    return torch.stack([(p.sum(1) * xs).sum(1), (p.sum(2) * ys).sum(1)], 1), p


def local_soft_argmax(logits, r=3):
    """Soft-argmax restricted to a (2r+1)^2 window around the arg-max (robust to tails)."""
    B, h, w = logits.shape
    out, conf = [], []
    for b in range(B):
        l = logits[b].float()
        idx = int(l.argmax())
        cy, cx = divmod(idx, w)
        y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
        win = l[y0:y1, x0:x1]
        p = (win - win.max()).exp()
        p = p / p.sum()
        ys = torch.arange(y0, y1, device=l.device, dtype=torch.float32)
        xs = torch.arange(x0, x1, device=l.device, dtype=torch.float32)
        out.append(torch.stack([hm_to_px((p.sum(0) * xs).sum()), hm_to_px((p.sum(1) * ys).sum())]))
        conf.append(float(l.flatten().softmax(0).view(h, w)[y0:y1, x0:x1].sum()))
    return torch.stack(out), conf


def gauss_target(pts_px, h, w, sigma):
    xs = hm_to_px(torch.arange(w, device=DEV, dtype=torch.float32))
    ys = hm_to_px(torch.arange(h, device=DEV, dtype=torch.float32))
    gx = torch.exp(-((xs[None] - pts_px[:, 0:1]) ** 2) / (2 * sigma ** 2))
    gy = torch.exp(-((ys[None] - pts_px[:, 1:2]) ** 2) / (2 * sigma ** 2))
    t = gy[:, :, None] * gx[:, None, :]
    return t / t.sum((1, 2), keepdim=True).clamp_min(1e-12)


# ----------------------------------------------------------------------------- stage configs
S1_OUT = (1024, 704)


AUG = {"scale": (0.85, 1.2), "shift": 0.08, "rot": 15}


def s1_matrix(img, rng=None):
    H, W = img.shape[1:]
    s0 = min(S1_OUT[0] / W, S1_OUT[1] / H)
    csrc = ((W - 1) / 2, (H - 1) / 2)
    cdst = ((S1_OUT[0] - 1) / 2, (S1_OUT[1] - 1) / 2)
    if rng is None:
        return affine(csrc, cdst, s0, 0, False)
    sc = s0 * rng.uniform(*AUG["scale"])
    sh = AUG["shift"]
    cdst = (cdst[0] + rng.uniform(-sh, sh) * S1_OUT[0], cdst[1] + rng.uniform(-sh, sh) * S1_OUT[1])
    return affine(csrc, cdst, sc, rng.uniform(-AUG["rot"], AUG["rot"]), rng.random() < 0.5)


def apply(M, p):
    return M[:, :2] @ np.asarray(p) + M[:, 2]


def make_batch(ids, cache, gt_half, rng):
    xs, pts = [], []
    for i in ids:
        img = cache[i]["img"].to(DEV)
        M = s1_matrix(img, rng)
        out = S1_OUT
        x = color_aug(warp(img, M, out), rng)
        xs.append(x)
        pts.append(apply(M, gt_half[i]))
    return normalize(torch.stack(xs)), torch.tensor(np.array(pts), dtype=torch.float32, device=DEV)


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(models, ids, cache, tta=True):
    """Return dict id -> predicted half-res point (mean over models x flips)."""
    res = {}
    for i in ids:
        img = cache[i]["img"].to(DEV)
        preds = []
        for flip in ([False, True] if tta else [False]):
            M = s1_matrix(img)
            if flip:
                M = np.array([[-1, 0, S1_OUT[0] - 1], [0, 1, 0]]) @ np.vstack([M, [0, 0, 1]])
            out = S1_OUT
            x = normalize(warp(img, M, out)[None])
            Ai = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
            for m in models:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = m(x.contiguous(memory_format=torch.channels_last))
                p, _ = local_soft_argmax(lg)
                p = p[0].cpu().numpy()
                preds.append(Ai[:, :2] @ p + Ai[:, 2])
        res[i] = np.mean(preds, 0)
    return res


# ----------------------------------------------------------------------------- training
def train_one(tr_ids, cache, gt_half, args, seed, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = HeatNet(args.arch).to(DEV).to(memory_format=torch.channels_last)
    bb = [p for n, p in m.named_parameters() if n.startswith("bb.")]
    hd = [p for n, p in m.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr}, {"params": hd, "lr": args.lr * 5}],
                            weight_decay=0.05)
    base = [g["lr"] for g in opt.param_groups]
    sigma = args.sigma
    t0 = time.time()
    for step in range(args.steps):
        fr = step / args.steps
        w = min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * fr))
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * w
        ids = rng.choice(tr_ids, args.bs, replace=True)
        x, pts = make_batch(ids, cache, gt_half, rng)
        m.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(x.contiguous(memory_format=torch.channels_last))
        lg = lg.float()
        pred, p = soft_argmax(lg)
        tgt = gauss_target(pts, lg.shape[1], lg.shape[2], sigma)
        kl = (tgt * (torch.log(tgt.clamp_min(1e-12)) - torch.log(p.clamp_min(1e-12)))).sum((1, 2)).mean()
        l1 = (pred - pts).abs().sum(1).mean()
        loss = kl + args.l1w * l1 / sigma
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        if log and (step % 250 == 0 or step == args.steps - 1):
            msg = f"  step {step} loss {loss.item():.4f} kl {kl.item():.3f} l1 {l1.item():.2f} {time.time()-t0:.0f}s"
            print(msg, file=log, flush=True)
    m.eval()
    return m


def rmse_full(pred_half, gt_full, ids):
    e = [half_to_full(pred_half[i]) - gt_full[i] for i in ids]
    e = np.array(e)
    return float(np.sqrt((e ** 2).mean())), np.sqrt((e ** 2).sum(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1])
    ap.add_argument("--arch", default="convnext_small", choices=["convnext_small"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--sigma", type=float, default=8.0)  # output pixels
    ap.add_argument("--l1w", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--centers", default="", choices=[""], help="compatibility only; empty, unused in stage 1")
    ap.add_argument("--full", action="store_true", help="train on all 80 instead of CV; predict test only")
    ap.add_argument("--prep", default="none", choices=["none"], help="half_cache only; no extra preprocessing")
    ap.add_argument("--res", type=float, default=1.0, choices=[1.0, 0.5], help="input scale relative to 1024x704")
    ap.add_argument("--val_every", type=int, default=0, choices=[0], help="compatibility only; no intermediate validation")
    ap.add_argument("--n_full", type=int, default=1)
    ap.add_argument("--regw", type=float, default=0.0, choices=[0.0], help="compatibility only; reg head retained, no reg loss")
    ap.add_argument("--only_fold", type=int, default=-1, choices=[-1], help="compatibility only; always run all CV folds")
    ap.add_argument("--own_frac", type=float, default=0.5, choices=[0.5], help="compatibility only; no external data")
    ap.add_argument("--ext_until", type=float, default=0.8, choices=[0.8], help="compatibility only; no external data")
    args = ap.parse_args()
    tag = args.tag or f"s{args.stage}_{args.arch}"
    global S1_OUT
    S1_OUT = (int(round(1024 * args.res / 32) * 32), int(round(704 * args.res / 32) * 32))
    os.makedirs(f"{ROOT}/out/{tag}", exist_ok=True)
    log = open(f"{ROOT}/logs/{tag}.log", "a")
    print(json.dumps({k: v for k, v in vars(args).items()}), file=log, flush=True)

    cache = load_cache()
    gt = pd.read_csv(f"{DATA}/fovea_localization_train_GT.csv")
    gt_full = {int(r.data): np.array([r.Fovea_X, r.Fovea_Y]) for r in gt.itertuples()}
    gt_half = {i: full_to_half(p) for i, p in gt_full.items()}
    train_ids = np.array(sorted(gt_full))
    test_ids = sorted(i for i in cache if i not in gt_full)

    if args.full:
        # train on all 80 labelled images (no OOF), predict test only
        preds = []
        for k in range(args.n_full):
            print(f"full model {k}", file=log, flush=True)
            m = train_one(train_ids, cache, gt_half, args, args.seed * 100 + 50 + k, log=log)
            preds.append(predict([m], test_ids, cache))
            torch.save(m.state_dict(), f"{ROOT}/out/{tag}/full{k}.pt")
            del m
            torch.cuda.empty_cache()
        test = {i: np.mean([p[i] for p in preds], 0).tolist() for i in test_ids}
        json.dump(test, open(f"{ROOT}/out/{tag}/test_full.json", "w"))
        print("DONE full", file=log, flush=True)
        return
    folds = np.random.default_rng(123).permutation(len(train_ids)) % args.folds
    oof, test_preds = {}, []
    for f in range(args.folds):
        tr, va = train_ids[folds != f], train_ids[folds == f]
        print(f"fold {f}", file=log, flush=True)
        m = train_one(tr, cache, gt_half, args, args.seed * 100 + f, log=log)
        pv = predict([m], va, cache)
        oof.update(pv)
        r, _ = rmse_full(pv, gt_full, va)
        print(f"fold {f} rmse {r:.2f}", file=log, flush=True)
        test_preds.append(predict([m], test_ids, cache))
        torch.save(m.state_dict(), f"{ROOT}/out/{tag}/fold{f}.pt")
        del m
        torch.cuda.empty_cache()
    r, per = rmse_full(oof, gt_full, train_ids)
    print(f"OOF rmse {r:.3f}  median dist {np.median(per):.2f}  max {per.max():.1f}", file=log, flush=True)
    test = {i: np.mean([tp[i] for tp in test_preds], 0) for i in test_ids}
    allc = {**{int(k): v.tolist() for k, v in oof.items()}, **{int(k): v.tolist() for k, v in test.items()}}
    json.dump(allc, open(f"{ROOT}/out/{tag}/centers.json", "w"))
    json.dump({"oof_rmse": r, "per_image_dist": dict(zip(map(int, train_ids), per.tolist()))},
              open(f"{ROOT}/out/{tag}/oof.json", "w"))
    print("DONE", file=log, flush=True)


if __name__ == "__main__":
    main()
