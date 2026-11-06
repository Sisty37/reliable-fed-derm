"""
E8 (protocol v0.12, Section 5c): randomized sibling-exposure crossover on fixed test images.

Starting point: the lesion-grouped re-splits with validation fold (splits/e1/resplit_lesion_seed{s}_val.csv),
where no lesion crosses train / val / test.

  * Target lesions: test lesions with >= 2 test images. One image per target lesion is drawn at random as
    its fixed TARGET image; the other images of that lesion are its SIBLINGS.
  * Target lesions are randomized to two groups G1 / G2, stratified by center x class; within each stratum
    lesions are sorted by sibling count and assigned in random pairs, so both groups get almost the same
    number of sibling images (training sets of the two arms have almost the same size).
  * Arm 1: siblings of G1 go to training (fold "train", same center), siblings of G2 are removed
    (fold "excluded"). Arm 2: the reverse. Target images, all other test images, validation and the
    original training images are identical in both arms.
  * Every target image is therefore evaluated once with its siblings in training (exposed) and once without
    (unexposed), by models trained with the same seed, recipe and almost the same number of images.
    The paired difference is a randomized estimate of the effect of sibling exposure on that image.

Output: splits/e8/sibexp_lesion_seed{s}_arm{1,2}.csv (same columns as the input; folds train/val/test/excluded),
splits/e8/sibexp_manifest_seed{s}.csv (target image, lesion, center, class, group, n_siblings),
splits/e8/e8_split_summary.csv.

Usage (from the repo folder):  python scripts/make_e8_splits.py
"""
import argparse
import os

import numpy as np
import pandas as pd


def make(df, seed):
    rng = np.random.default_rng(20261107 + seed)
    test = df[df.fold == "test"]
    size = test.groupby("lesion_id").size()
    targets = sorted(size[size >= 2].index)
    rows = []
    for les in targets:
        imgs = sorted(test.image[test.lesion_id == les])
        t = imgs[int(rng.integers(len(imgs)))]
        r = test[test.image == t].iloc[0]
        rows.append({"lesion_id": les, "target_image": t, "center": int(r.center), "target": int(r.target),
                     "n_siblings": len(imgs) - 1, "siblings": ";".join(i for i in imgs if i != t)})
    man = pd.DataFrame(rows)
    man["group"] = 0
    for _, g in man.groupby(["center", "target"]):
        g = g.sample(frac=1, random_state=int(rng.integers(2**31))).sort_values("n_siblings", kind="stable")
        idx = g.index.to_numpy()
        first = int(rng.integers(2))                       # random group for the first lesion of each pair
        for k in range(0, len(idx), 2):
            pair = idx[k:k + 2]
            order = [1, 2] if rng.random() < 0.5 else [2, 1]
            if len(pair) == 1:
                order = [1 + first]
            for i, gr in zip(pair, order):
                man.loc[i, "group"] = gr
    arms = {}
    for arm in (1, 2):
        d = df.copy()
        for _, r in man.iterrows():
            sib = r.siblings.split(";")
            d.loc[d.image.isin(sib), "fold"] = "train" if r.group == arm else "excluded"
        arms[arm] = d
    return man, arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join("splits", "e1"))
    ap.add_argument("--out", default=os.path.join("splits", "e8"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summ = []
    for s in a.seeds:
        df = pd.read_csv(os.path.join(a.src, f"resplit_lesion_seed{s}_val.csv"))
        man, arms = make(df, s)
        man.drop(columns="siblings").to_csv(os.path.join(a.out, f"sibexp_manifest_seed{s}.csv"), index=False)
        for arm, d in arms.items():
            d.to_csv(os.path.join(a.out, f"sibexp_lesion_seed{s}_arm{arm}.csv"), index=False)
            tr = d[d.fold == "train"]
            te = d[d.fold == "test"]
            # integrity checks: targets stay in test; no target lesion has a sibling in training of the other arm's group
            exposed = set(man.lesion_id[man.group == arm])
            assert set(man.target_image) <= set(te.image)
            assert set(tr.lesion_id) & set(man.lesion_id) == exposed
            assert set(d.lesion_id[d.fold == "val"]) & set(man.lesion_id) == set()
            summ.append({"seed": s, "arm": arm, "train_images": len(tr), "test_images": len(te),
                         "excluded_images": int((d.fold == "excluded").sum()),
                         "target_lesions": len(man), "exposed_lesions": len(exposed),
                         "sibling_images_added": int(man.n_siblings[man.group == arm].sum()),
                         "targets_per_center": " ".join(str(int((man.center == c).sum())) for c in range(6))})
    pd.DataFrame(summ).to_csv(os.path.join(a.out, "e8_split_summary.csv"), index=False)
    print(pd.DataFrame(summ).to_string(index=False))


if __name__ == "__main__":
    main()
