"""Step 2: cross-validation split.

StratifiedGroupKFold over exact-duplicate groups so that identical images
never straddle the train/validation boundary (otherwise validation accuracy
is inflated by memorisation).
"""
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
SEED = 20261011
N_FOLDS = 5


def main():
    y = np.load(os.path.join(ART, "train_labels.npy"))
    g = np.load(os.path.join(ART, "train_group.npy"))
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.full(len(y), -1, dtype=np.int64)
    for k, (_, va) in enumerate(splitter.split(np.zeros(len(y)), y, g)):
        fold[va] = k
    assert (fold >= 0).all()

    # leakage check: no group may appear in two folds
    for gid in np.unique(g):
        assert len(np.unique(fold[g == gid])) == 1, gid

    np.save(os.path.join(ART, "fold.npy"), fold)
    report = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "scheme": "StratifiedGroupKFold on exact-duplicate pixel groups",
        "fold_sizes": {int(k): int((fold == k).sum()) for k in range(N_FOLDS)},
        "fold_class_counts": {
            int(k): np.bincount(y[fold == k], minlength=6).tolist() for k in range(N_FOLDS)
        },
        "group_leak_across_folds": 0,
    }
    with open(os.path.join(ART, "folds.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
