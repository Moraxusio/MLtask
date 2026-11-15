"""Shared training utilities: dataset, augmentation, model factory."""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
WEIGHTS = os.path.join(ROOT, "weights")
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Only the three architectures contributing to the actual submission.
ARCH_WEIGHTS = {
    "resnet34": "resnet34.a1_in1k.safetensors",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k.safetensors",
    "tf_efficientnet_b0": "tf_efficientnet_b0.ns_jft_in1k.safetensors",
}


def build_model(arch, num_classes=6, drop_rate=0.2, drop_path_rate=0.1):
    """timm model with locally cached ImageNet weights (no network access)."""
    import safetensors.torch as st
    import timm

    kwargs = dict(pretrained=False, num_classes=num_classes, drop_rate=drop_rate)
    try:
        model = timm.create_model(arch, drop_path_rate=drop_path_rate, **kwargs)
    except TypeError:
        model = timm.create_model(arch, **kwargs)

    path = os.path.join(WEIGHTS, ARCH_WEIGHTS[arch])
    sd = st.load_file(path)
    head_prefixes = ("fc.", "classifier.", "head.")
    # Drop any tensor whose shape disagrees with the target model: these are
    # the classifier/norm layers of the source task, which must be relearned.
    tgt = model.state_dict()
    sd = {k: v for k, v in sd.items()
          if not k.startswith(head_prefixes)
          and k in tgt and tgt[k].shape == v.shape}
    result = model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(head_prefixes)]
    if result.unexpected_keys or missing:
        raise RuntimeError(
            f"{arch}: missing={missing[:5]} unexpected={result.unexpected_keys[:5]}")
    loaded = sum(v.numel() for v in sd.values())
    total = sum(v.numel() for k, v in tgt.items() if not k.startswith(head_prefixes))
    if loaded < 0.9 * total:
        raise RuntimeError(
            f"{arch}: only {loaded/1e6:.1f}M of {total/1e6:.1f}M backbone "
            "parameters matched; weight file likely mismatched")
    return model


# ---------------------------------------------------------------- augmentation
def _rand_affine_grid(n, img, max_rot, max_trans, max_scale, max_shear, device, gen):
    """Batched affine sampling grid; one random transform per image."""
    ang = (torch.rand(n, generator=gen, device=device) * 2 - 1) * math.radians(max_rot)
    sc = 1.0 + (torch.rand(n, generator=gen, device=device) * 2 - 1) * max_scale
    shr = (torch.rand(n, generator=gen, device=device) * 2 - 1) * math.radians(max_shear)
    tx = (torch.rand(n, generator=gen, device=device) * 2 - 1) * max_trans
    ty = (torch.rand(n, generator=gen, device=device) * 2 - 1) * max_trans
    cos, sin = torch.cos(ang), torch.sin(ang)
    tan = torch.tan(shr)
    # rotation * shear, then inverse scale (grid_sample uses inverse mapping)
    a = (cos + sin * tan) / sc
    b = sin / sc
    c = (-sin + cos * tan) / sc
    d = cos / sc
    theta = torch.stack(
        [torch.stack([a, b, tx], 1), torch.stack([c, d, ty], 1)], 1
    )
    return F.affine_grid(theta, img.shape, align_corners=False)


def augment(x, gen, cfg):
    """GPU augmentation on [N,1,H,W] float tensor in [0,1]."""
    n = x.shape[0]
    dev = x.device
    # horizontal flip
    flip = torch.rand(n, generator=gen, device=dev) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    # affine
    grid = _rand_affine_grid(
        n, x, cfg["rot"], cfg["trans"], cfg["scale"], cfg["shear"], dev, gen
    )
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                      align_corners=False)
    # brightness / contrast
    br = 1.0 + (torch.rand(n, 1, 1, 1, generator=gen, device=dev) * 2 - 1) * cfg["bright"]
    ct = 1.0 + (torch.rand(n, 1, 1, 1, generator=gen, device=dev) * 2 - 1) * cfg["contrast"]
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - mean) * ct + mean) * br
    x = x.clamp(0, 1)
    # random erasing
    if cfg["erase_p"] > 0:
        x = random_erase(x, gen, cfg["erase_p"], cfg["erase_max"])
    return x


def random_erase(x, gen, p, max_area):
    n, _, h, w = x.shape
    do = torch.rand(n, generator=gen, device=x.device) < p
    if not do.any():
        return x
    area = (torch.rand(n, generator=gen, device=x.device) * (max_area - 0.02) + 0.02)
    ratio = torch.exp(
        (torch.rand(n, generator=gen, device=x.device) * 2 - 1) * math.log(3.0)
    )
    eh = (torch.sqrt(area * h * w * ratio)).clamp(1, h - 1).long()
    ew = (torch.sqrt(area * h * w / ratio)).clamp(1, w - 1).long()
    y0 = (torch.rand(n, generator=gen, device=x.device) * (h - eh).float()).long()
    x0 = (torch.rand(n, generator=gen, device=x.device) * (w - ew).float()).long()
    rows = torch.arange(h, device=x.device).view(1, h, 1)
    cols = torch.arange(w, device=x.device).view(1, 1, w)
    mask = (
        (rows >= y0.view(-1, 1, 1))
        & (rows < (y0 + eh).view(-1, 1, 1))
        & (cols >= x0.view(-1, 1, 1))
        & (cols < (x0 + ew).view(-1, 1, 1))
    ) & do.view(-1, 1, 1)
    noise = torch.rand(x.shape, generator=gen, device=x.device)
    return torch.where(mask.unsqueeze(1), noise, x)


def to_model_input(x, size):
    """[N,1,h,w] in [0,1] -> [N,3,size,size] ImageNet-normalised."""
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
        x = x.clamp(0, 1)
    x = x.repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ------------------------------------------------------------------- mix / cut
def mixup_cutmix(x, y, gen, alpha_mix, alpha_cut, prob, num_classes):
    """Returns mixed input and soft targets."""
    y1 = F.one_hot(y, num_classes).float()
    if prob <= 0 or torch.rand(1, generator=gen, device=x.device).item() > prob:
        return x, y1
    perm = torch.randperm(x.shape[0], generator=gen, device=x.device)
    use_cut = torch.rand(1, generator=gen, device=x.device).item() < 0.5
    if use_cut and alpha_cut > 0:
        lam = float(np.random.beta(alpha_cut, alpha_cut))
        h, w = x.shape[-2:]
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        cy = int(torch.randint(h, (1,), generator=gen, device=x.device).item())
        cx = int(torch.randint(w, (1,), generator=gen, device=x.device).item())
        y0, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x0, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x = x.clone()
        x[:, :, y0:y2, x0:x2] = x[perm][:, :, y0:y2, x0:x2]
        lam = 1 - ((y2 - y0) * (x2 - x0) / (h * w))
    elif alpha_mix > 0:
        lam = float(np.random.beta(alpha_mix, alpha_mix))
        x = lam * x + (1 - lam) * x[perm]
    else:
        return x, y1
    return x, lam * y1 + (1 - lam) * y1[perm]


def soft_ce(logits, target, smoothing=0.0):
    if smoothing > 0:
        target = target * (1 - smoothing) + smoothing / target.shape[1]
    return -(target * F.log_softmax(logits, dim=1)).sum(1).mean()


class PixelDataset(Dataset):
    """Serves raw uint8 48x48 images; all augmentation happens on GPU."""

    def __init__(self, pixels, labels=None, indices=None):
        self.pixels = pixels
        self.labels = labels
        self.indices = np.arange(len(pixels)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        img = torch.from_numpy(self.pixels[j].copy())
        if self.labels is None:
            return img, 0
        return img, int(self.labels[j])


class ModelEma:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def copy_to(self, model):
        model.load_state_dict({k: v for k, v in self.shadow.items()}, strict=True)


def load_data():
    tr = np.load(os.path.join(ART, "train_pixels.npy"))
    te = np.load(os.path.join(ART, "test_pixels.npy"))
    y = np.load(os.path.join(ART, "train_labels.npy"))
    fold = np.load(os.path.join(ART, "fold.npy"))
    with open(os.path.join(ART, "file_index.json")) as f:
        idx = json.load(f)
    return tr, te, y, fold, idx
