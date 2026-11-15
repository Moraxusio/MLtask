"""Fixed actual-v4 ensemble: 25 U-Net + 5 ResNet34x2 + 5 EfficientNet-B3x2.

No family/threshold search and no checkpoint globbing. Default is a file-only
preflight; --evaluate-oof is CPU-only; --infer explicitly enables CUDA inference
and CSV export. See README.md for evidence and historical training limitations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
UNET_TAGS = ["s0", "s1", "s2", "s3"]
UNET_FULL = ["full_s10", "full_s11", "full_s12", "full_s13", "full_s14"]
TIMM = {
    "r34x2": ("resnet34.a1_in1k", 2.0),
    "eb3x2": ("efficientnet_b3.ra2_in1k", 2.0),
}
THRESHOLD = 0.36
IMAGE_IDS = [str(i) for i in range(1, 21)]


def checkpoint_paths(work):
    """Exact filenames and original model accumulation order; never scan work/."""
    work = Path(work)
    return {
        "unet": [work / f"fold{k}_{t}.pt" for t in UNET_TAGS for k in range(5)]
                + [work / f"fold0_{t}.pt" for t in UNET_FULL],
        **{t: [work / f"fold{k}_{t}_s0.pt" for k in range(5)] for t in TIMM},
    }


def check_checkpoints(work):
    paths = checkpoint_paths(work)
    missing = [str(p) for group in paths.values() for p in group
               if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing/empty required v4 checkpoints:\n" + "\n".join(missing))
    print("Required checkpoint files:", {k: len(v) for k, v in paths.items()})
    return paths


def load_oof(path, shape=(20, 584, 565)):
    a = np.load(path, allow_pickle=False)
    if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError(f"Invalid OOF probabilities: {path}")
    return a


def assemble(tag, work):
    """Original seed-0 five-fold assembly, in memory (does not overwrite caches)."""
    work = Path(work)
    cached = work / f"oof_{tag}_s0.npy"
    parts = [work / f"partial_{tag}_s0_f{k}.npy" for k in range(5)]
    oof = load_oof(cached) if cached.is_file() else None
    if all(p.is_file() for p in parts):
        va = np.array_split(np.random.RandomState(0).permutation(20), 5)
        assembled = np.zeros((20, 584, 565), np.float32)
        for k, p in enumerate(parts):
            assembled[va[k]] = load_oof(p, (4, 584, 565))
        if oof is not None and not np.array_equal(oof, assembled):
            raise ValueError(f"Cached and partial OOF disagree: {tag}")
        return assembled
    if oof is None:
        raise FileNotFoundError(f"Need {cached} or all five partial files")
    return oof


def evaluate_oof(work, seg):
    from predict import official_mask

    def dice(p, g):
        return 2 * (p * g).sum() / (p.sum() + g.sum() + 1e-9)

    u = np.mean([load_oof(Path(work) / f"oof_{t}.npy") for t in UNET_TAGS], 0)
    # Preserve the original float32 family accumulation order and arithmetic.
    p = (u + assemble("r34x2", work) + assemble("eb3x2", work)) / 3
    pipeline, native = [], []
    for j, i in enumerate(IMAGE_IDS):
        with Image.open(Path(seg) / "train" / "label" / f"{i}.jpg") as im:
            g = np.array(im.convert("L"))
        if g.shape != (584, 565):
            raise ValueError(f"Unexpected label shape for ID {i}: {g.shape}")
        mask = p[j] > THRESHOLD
        pipeline.append(dice(official_mask((mask * 255).astype(np.uint8)), official_mask(g)))
        native.append(dice(mask, g > 127))
    result = {"weights": {n: 1 for n in ("unet", *TIMM)}, "threshold": THRESHOLD,
              "oof_pipeline_dice": float(np.mean(pipeline)),
              "oof_native_dice": float(np.mean(native))}
    print(json.dumps(result, indent=2))
    return result


def infer(work, seg, output_dir):
    # Check all 35 paths before importing/allocating CUDA models. Extra files are ignored.
    paths = check_checkpoints(work)
    import torch
    from train import UNet, load_img, predict
    from train2 import TimmUNet, predict2
    from segmentation.segmentation_to_csv import turn_to_str, save_to_csv

    seg, output_dir = Path(seg), Path(output_dir)
    images = [seg / "test" / "image" / f"{i}.jpg" for i in IMAGE_IDS]
    for path in images:
        with Image.open(path) as im:
            if im.size != (565, 584):
                raise ValueError(f"Unexpected test image size: {path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    od = output_dir / "image"
    od.mkdir(exist_ok=True)
    fam_models = {}
    for n, group in paths.items():
        fam_models[n] = []
        for path in group:
            m = (UNet() if n == "unet" else TimmUNet(TIMM[n][0], pretrained=False)).cuda()
            m.load_state_dict(torch.load(path, map_location="cuda", weights_only=True), strict=True)
            m.eval()
            fam_models[n].append(m)
    for i, path in zip(IMAGE_IDS, images):
        x, fov = load_img(str(path))
        acc = 0
        for n, ms in fam_models.items():
            p = predict(ms, x) if n == "unet" else predict2(ms, x, TIMM[n][1])
            acc = acc + p
        prob = acc / 3 * fov
        Image.fromarray(np.where(prob > THRESHOLD, 255, 0).astype(np.uint8)).save(od / f"{i}.png")
    # Numeric ID order is deterministic; source converter's RLE math is unchanged.
    names = [f"{i}.png" for i in IMAGE_IDS]
    save_to_csv(names, turn_to_str([od / n for n in names]), output_dir / "submission.csv")
    record = {"weights": {n: 1 for n in fam_models}, "threshold": THRESHOLD,
              "n_models": {n: len(ms) for n, ms in fam_models.items()},
              "checkpoints": {n: [p.name for p in group] for n, group in paths.items()}}
    (output_dir / "final_v4.json").write_text(json.dumps(record, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=ROOT / "work")
    ap.add_argument("--seg", type=Path, default=ROOT / "segmentation")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "v4")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="check 35 exact nonempty filenames (default)")
    mode.add_argument("--evaluate-oof", action="store_true", help="CPU: evaluate saved OOF at fixed .36, no search")
    mode.add_argument("--infer", action="store_true", help="CUDA: generate masks and submission using fixed v4")
    a = ap.parse_args()
    if a.evaluate_oof:
        evaluate_oof(a.work, a.seg)
    elif a.infer:
        infer(a.work, a.seg, a.output_dir)
    else:
        check_checkpoints(a.work)


if __name__ == "__main__":
    main()
