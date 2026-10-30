"""
E1 (protocol v0.10, Section 3): add the lesion-grouped validation fold to the locked re-splits.

For each re-split file splits/resplit_{image,lesion}_seed{s}.csv, 10% of the training lesions of each
center are moved to fold "val" (all their training images), in both conditions. Lesions with
training images in more than one center are never chosen. The test fold is not
touched. Validation lesions are used later for temperature scaling (and checkpoint choice if needed);
they are not used for training.

Output: splits/e1/resplit_{image,lesion}_seed{s}_val.csv (+ e1_split_summary.csv). The locked files in
splits/ are only read, never changed.

Usage (from the repo folder):  python scripts\\make_e1_splits.py            # seeds 0 1 2
"""
import argparse
import os

import numpy as np
import pandas as pd

VAL_FRAC = 0.10


def add_val(df, seed):
    df = df.copy()
    rng = np.random.default_rng(seed)
    train = df[df.fold == "train"]
    multi = set(train.groupby("lesion_id").center.nunique().loc[lambda x: x > 1].index)
    for c in sorted(df.center.unique()):
        tr = df[(df.fold == "train") & (df.center == c)]
        # lesions with training images in two centers stay in training (as in the re-splits)
        lesions = np.array(sorted(set(tr.lesion_id) - multi))
        n_val = int(round(VAL_FRAC * len(lesions)))
        val_les = set(rng.choice(lesions, size=n_val, replace=False))
        df.loc[tr.index[tr.lesion_id.isin(val_les)], "fold"] = "val"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", default="splits")
    ap.add_argument("--out", default=os.path.join("splits", "e1"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for s in a.seeds:
        for kind in ("image", "lesion"):
            src = os.path.join(a.split_dir, f"resplit_{kind}_seed{s}.csv")
            df = pd.read_csv(src)
            assert len(df) == 23247 and set(df.fold) == {"train", "test"}, f"unexpected content in {src}"
            v = add_val(df, seed=1000 + s)                  # own stream, independent of the split seed
            v.to_csv(os.path.join(a.out, f"resplit_{kind}_seed{s}_val.csv"), index=False)
            tr, va, te = (v[v.fold == f] for f in ("train", "val", "test"))
            tr_les = set(tr.lesion_id)
            rows.append({
                "split": kind, "seed": s, "train": len(tr), "val": len(va), "test": len(te),
                "val_lesions": va.lesion_id.nunique(),
                "val_lesions_also_in_train": len(set(va.lesion_id) & tr_les),
                "test_images_lesion_in_train": round(float(te.lesion_id.isin(tr_les).mean()), 4),
                "test_images_lesion_in_train_or_val": round(float(
                    te.lesion_id.isin(tr_les | set(va.lesion_id)).mean()), 4),
                "val_per_center": " ".join(str((va.center == c).sum()) for c in range(6)),
            })
            print(rows[-1])
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "e1_split_summary.csv"), index=False)
    print(f"\nsaved {len(rows)} files to {a.out}")


if __name__ == "__main__":
    main()
