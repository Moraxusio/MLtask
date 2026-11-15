"""Write submission CSV from test predictions (half-res coords) and validate format.

Usage: make_sub.py OUT.csv file1.json [file2.json ...]
Each json maps image id -> [x_half, y_half]; centers.json (OOF+test) or test_full.json both work.
Predictions are averaged with equal weight.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.abspath(os.environ.get("MLTASK3_DATA_DIR", f"{ROOT}/data/detection"))


def half_to_full(p):
    return (np.asarray(p, dtype=np.float64) + 0.5) * 2 - 0.5


def main(out_csv, paths):
    if not paths:
        raise ValueError("Provide at least one model-prediction JSON")
    cs = []
    for path in paths:
        with open(path) as f:
            cs.append({int(k): np.asarray(v, dtype=np.float64)
                       for k, v in json.load(f).items()})
    sub = pd.read_csv(f"{DATA}/sample_submission.csv", usecols=["ImageID"])
    if len(sub) != 40 or sub.ImageID.isna().any() or not sub.ImageID.is_unique:
        raise ValueError("Template must contain 40 unique ImageIDs")
    vals = []
    for iid in sub.ImageID:
        n, kind, ax = iid.split("_")
        if kind != "Fovea" or ax not in ("X", "Y"):
            raise ValueError(f"Invalid template ImageID: {iid}")
        n = int(n)
        points = [c[n] for c in cs]
        if any(p.shape != (2,) or not np.isfinite(p).all() for p in points):
            raise ValueError(f"Invalid model coordinates for image {n}")
        p = half_to_full(np.mean(points, 0))
        with Image.open(f"{DATA}/test/{n:04d}.jpg") as image:
            W, H = image.size
        if not (0 <= p[0] < W and 0 <= p[1] < H):
            raise ValueError(f"Out-of-bounds coordinates: {(n, p, W, H)}")
        vals.append(p[0] if ax == "X" else p[1])
    sub["value"] = vals
    assert sub.value.notna().all() and len(sub) == 40
    sub.to_csv(out_csv, index=False)
    print("written", out_csv, len(sub), "from", paths)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
