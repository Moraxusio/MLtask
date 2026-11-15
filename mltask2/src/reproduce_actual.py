"""Rebuild the historical fixed ensemble and training-pixel duplicate override.

No training, inference, test labels, submission template, or selected-probability
cache is used. Input artifacts are read-only; the only file written is --out.
The fixed recipe comes from historical selection.json, not a new search.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

CLASSES = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
VIEWS = {
    "p_cnt_224_30": "tta",
    "p_r34_224_30": "tta",
    "eb0_224_30": "plain",
}
# Preserve the original greedy addition order as well as its effective weights.
PICKS = ("p_cnt_224_30", "p_r34_224_30", "p_cnt_224_30", "eb0_224_30")
N_FOLDS = 5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_prob(p, rows, description):
    require(p.shape == (rows, len(CLASSES)), f"{description}: wrong shape")
    require(np.isfinite(p).all() and (p >= 0).all(),
            f"{description}: invalid probability values")
    require(np.allclose(p.sum(1), 1, atol=1e-5),
            f"{description}: probabilities are not normalised")


def collect(art, tag, y, fold, n_test, fingerprints):
    """Same float64 accumulation and fold order as original select.collect."""
    oof = {v: np.zeros((len(y), len(CLASSES)), dtype=np.float64)
           for v in ("plain", "tta")}
    test = {v: np.zeros((n_test, len(CLASSES)), dtype=np.float64)
            for v in ("plain", "tta")}
    coverage = np.zeros(len(y), dtype=np.int64)
    for f in range(N_FOLDS):
        path = art / f"{tag}_fold{f}_preds.npz"
        fingerprints[path.name] = sha256(path)
        with np.load(path, allow_pickle=False) as d:
            vi = d["va_idx"]
            require(np.array_equal(vi, np.flatnonzero(fold == f)),
                    f"{path.name}: validation indices differ from fold.npy")
            coverage[vi] += 1
            for v in ("plain", "tta"):
                va, te = d[f"va_{v}"], d[f"te_{v}"]
                check_prob(va, len(vi), f"{path.name}/va_{v}")
                check_prob(te, n_test, f"{path.name}/te_{v}")
                oof[v][vi] = va
                test[v] += te
    require((coverage == 1).all(), f"{tag}: OOF coverage must be exactly once")
    for v in test:
        test[v] /= N_FOLDS
    metrics = {f"oof_accuracy_{v}": float((oof[v].argmax(1) == y).mean())
               for v in oof}
    metrics["fixed_view"] = VIEWS[tag]
    return oof[VIEWS[tag]], test[VIEWS[tag]], metrics


def combine(probabilities):
    # No class-prior adjustment (historical power = 0); no extra normalisation.
    return np.exp(sum(np.log(probabilities[tag] + 1e-9) for tag in PICKS)
                  / len(PICKS))


def override_duplicates(probs, train_pixels, test_pixels, y, min_votes=1):
    """Original dup_override rule: MD5 raw pixels, unanimous labels, else skip.

    Counts are calculated from the supplied training data, never sample IDs.
    """
    require(min_votes >= 1, "min_votes must be positive")
    h = lambda arr: hashlib.md5(arr.tobytes()).hexdigest()
    table = defaultdict(list)
    for i in range(len(train_pixels)):
        table[h(train_pixels[i])].append(i)
    pred = probs.argmax(1).copy()
    stats = dict(exact_training_twin_rows=0, overridden_rows=0,
                 already_agreeing_rows=0, conflicting_label_rows=0,
                 below_min_votes_rows=0)
    for j in range(len(test_pixels)):
        group = table.get(h(test_pixels[j]))
        if not group:
            continue
        stats["exact_training_twin_rows"] += 1
        vals, counts = np.unique(y[group], return_counts=True)
        if len(vals) > 1:
            stats["conflicting_label_rows"] += 1
            continue
        if counts[0] < min_votes:
            stats["below_min_votes_rows"] += 1
            continue
        lab = int(vals[0])
        if lab == pred[j]:
            stats["already_agreeing_rows"] += 1
        else:
            stats["overridden_rows"] += 1
            pred[j] = lab
    stats["training_duplicate_groups_with_conflicting_labels"] = sum(
        len(np.unique(y[group])) > 1 for group in table.values())
    return pred, stats


def oof_metrics(probs, y):
    pred = probs.argmax(1)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = cm.diagonal()
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(CLASSES)), where=denom != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(len(CLASSES)),
                       where=cm.sum(1) != 0)
    return dict(accuracy=float((pred == y).mean()), macro_f1=float(f1.mean()),
                per_class_recall=dict(zip(CLASSES, recall.tolist())),
                confusion_matrix=cm.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    art, out = args.artifacts.resolve(), args.out.resolve()
    require(art != out and art not in out.parents,
            "Output must be outside the read-only artifacts directory")
    require(out.suffix == ".csv", "Output must be a CSV file")
    fingerprints = {}
    for name in ("file_index.json", "train_pixels.npy", "test_pixels.npy",
                 "train_labels.npy", "fold.npy"):
        fingerprints[name] = sha256(art / name)
    with (art / "file_index.json").open(encoding="utf-8") as f:
        index = json.load(f)
    require(index["classes"] == CLASSES, "Class order does not match the recipe")
    train_pixels = np.load(art / "train_pixels.npy", mmap_mode="r", allow_pickle=False)
    test_pixels = np.load(art / "test_pixels.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(art / "train_labels.npy", allow_pickle=False)
    fold = np.load(art / "fold.npy", allow_pickle=False)
    require(y.ndim == 1 and np.issubdtype(y.dtype, np.integer)
            and ((y >= 0) & (y < len(CLASSES))).all(), "Invalid training labels")
    require(fold.shape == y.shape and set(np.unique(fold)) == set(range(N_FOLDS)),
            "Invalid fold assignments")
    require(len(index["train"]) == len(y), "Training index length mismatch")
    for pixels, n in ((train_pixels, len(y)), (test_pixels, len(index["test"]))):
        require(pixels.dtype == np.uint8 and pixels.shape == (n, 48, 48),
                "Pixel cache must contain raw uint8 48x48 images")
    require(all(name.split("/")[0] == CLASSES[int(lab)]
                for name, lab in zip(index["train"], y)),
            "Training labels do not match class directories in the index")
    names = index["test"]
    require(names == sorted(names) and len(names) == len(set(names)),
            "Test IDs must follow the original sorted, unique file-index order")
    require(all(not any(c in name for c in ",\r\n") for name in names),
            "Test IDs cannot contain CSV delimiters")
    oof, test, per_model = {}, {}, {}
    for tag in VIEWS:
        oof[tag], test[tag], per_model[tag] = collect(
            art, tag, y, fold, len(test_pixels), fingerprints)
    ensemble_oof, ensemble_test = combine(oof), combine(test)
    base = ensemble_test.argmax(1)
    pred, duplicates = override_duplicates(ensemble_test, train_pixels, test_pixels, y)
    # Exact original CSV representation: header, sorted IDs, integer labels, LF.
    content = "ID,Emotion\n" + "".join(
        f"{name},{int(p)}\n" for name, p in zip(names, pred))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(content.encode("utf-8"))
    report = dict(
        recipe="historical_log_ensemble_training_duplicate_override_v1",
        picks=list(PICKS), fixed_views=VIEWS, prior_correction_power=0.0,
        train_rows=len(y), test_rows=len(names), per_model=per_model,
        ensemble_oof=oof_metrics(ensemble_oof, y), duplicate_override=duplicates,
        prediction_counts_before=np.bincount(base, minlength=len(CLASSES)).tolist(),
        prediction_counts_after=np.bincount(pred, minlength=len(CLASSES)).tolist(),
        classes=CLASSES, input_sha256=fingerprints, output_sha256=sha256(out),
        output_bytes=len(content.encode("utf-8")),
        evaluation_scope="training OOF only; no test accuracy is measured")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
