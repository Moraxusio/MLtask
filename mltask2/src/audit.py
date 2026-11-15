"""Step 1: data overview / audit.

Builds a pixel cache and reports class balance, duplicate structure,
label conflicts and train-test leakage. All outputs are deterministic.
"""
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "fer_data", "fer_data")
ART = os.path.join(ROOT, "artifacts")
CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_split():
    train_files, train_labels = [], []
    for cls in CLASSES:
        d = os.path.join(DATA, "train", cls)
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(".jpg"):
                train_files.append(os.path.join(cls, name))
                train_labels.append(CLASS_TO_IDX[cls])
    test_files = sorted(
        n for n in os.listdir(os.path.join(DATA, "test")) if n.lower().endswith(".jpg")
    )
    return train_files, np.asarray(train_labels, dtype=np.int64), test_files


def read_pixels(paths, base):
    sizes, modes = Counter(), Counter()
    out = np.zeros((len(paths), 48, 48), dtype=np.uint8)
    for i, rel in enumerate(paths):
        with Image.open(os.path.join(base, rel)) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
            arr = np.asarray(im.convert("L"), dtype=np.uint8)
        if arr.shape != (48, 48):
            arr = np.asarray(
                Image.fromarray(arr).resize((48, 48), Image.BICUBIC), dtype=np.uint8
            )
        out[i] = arr
    return out, sizes, modes


def main():
    os.makedirs(ART, exist_ok=True)
    train_files, train_labels, test_files = load_split()

    tr_px, tr_sizes, tr_modes = read_pixels(train_files, os.path.join(DATA, "train"))
    te_px, te_sizes, te_modes = read_pixels(test_files, os.path.join(DATA, "test"))
    np.save(os.path.join(ART, "train_pixels.npy"), tr_px)
    np.save(os.path.join(ART, "test_pixels.npy"), te_px)
    np.save(os.path.join(ART, "train_labels.npy"), train_labels)
    with open(os.path.join(ART, "file_index.json"), "w") as f:
        json.dump({"train": train_files, "test": test_files, "classes": CLASSES}, f)

    # exact duplicates by pixel hash
    def hashes(px):
        return [hashlib.md5(px[i].tobytes()).hexdigest() for i in range(len(px))]

    tr_h, te_h = hashes(tr_px), hashes(te_px)
    groups = defaultdict(list)
    for i, h in enumerate(tr_h):
        groups[h].append(i)
    dup_groups = {h: idx for h, idx in groups.items() if len(idx) > 1}
    conflicting = 0
    for idx in dup_groups.values():
        if len(set(train_labels[idx].tolist())) > 1:
            conflicting += 1
    te_set = set(te_h)
    overlap = sum(1 for h in tr_h if h in te_set)

    # group id: exact-duplicate group, used later to prevent CV leakage
    group_id = np.empty(len(tr_px), dtype=np.int64)
    for gid, (_, idx) in enumerate(sorted(groups.items())):
        group_id[idx] = gid
    np.save(os.path.join(ART, "train_group.npy"), group_id)

    # intensity statistics (used for normalisation sanity, not for training)
    report = {
        "n_train": int(len(tr_px)),
        "n_test": int(len(te_px)),
        "classes": CLASSES,
        "class_counts": {c: int((train_labels == i).sum()) for i, c in enumerate(CLASSES)},
        "class_ratio": {
            c: round(float((train_labels == i).mean()), 4) for i, c in enumerate(CLASSES)
        },
        "imbalance_max_over_min": round(
            float(np.bincount(train_labels).max() / np.bincount(train_labels).min()), 3
        ),
        "train_image_sizes": {str(k): v for k, v in tr_sizes.items()},
        "train_image_modes": dict(tr_modes),
        "test_image_sizes": {str(k): v for k, v in te_sizes.items()},
        "exact_duplicate_groups": len(dup_groups),
        "exact_duplicate_extra_rows": int(sum(len(v) - 1 for v in dup_groups.values())),
        "duplicate_groups_with_conflicting_labels": conflicting,
        "train_rows_also_present_in_test": int(overlap),
        "unique_groups": int(len(groups)),
        "pixel_mean_0_1": round(float(tr_px.mean() / 255.0), 4),
        "pixel_std_0_1": round(float(tr_px.std() / 255.0), 4),
    }
    with open(os.path.join(ART, "audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
