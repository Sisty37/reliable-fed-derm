"""
E0b (protocol v0.10): build the three data pipelines of the mother paper (FedHybridXAI, Natha et al. 2026)
on the HAM10000 part of Fed-ISIC2019 (centers 1, 2, 3, 5), for seeds 0, 1, 2.

  arm a  "as described in the text": 60/20/20 split by lesion (GroupShuffleSplit, p. 293),
         then class balancing of the training set only to the Table 3 targets (p. 294)
  arm b  "as implied by the tables": the whole dataset is balanced to the Table 3 targets
         (22,520 rows), then an image-level stratified 70/15/15 split (test share from Table 9)
  arm c  image-level stratified 60/20/20 split, then the same training-only balancing as arm a
         (a and c differ only in lesion grouping)

Balancing (Table 3): the majority class (NV) is undersampled without replacement; every other class
gets extra copies made with label-preserving flips and 90-degree rotations. ISIC 2019 labels HAM10000's akiec images as AK or SCC; they are merged back into one akiec class
(code 3), as in the mother paper's 7 classes. Each row keeps its
source image (`image`) and a transform id (`tf`: 0 original, 1 h-flip, 2 v-flip, 3 rot90,
4 rot180, 5 rot270), so copies can be traced back to their source.
Training rows are split over 3 label-skewed clients with a Dirichlet(alpha = 0.5) class partition
(the mother paper, p. 298, does not say how its clients were built).

Output: splits/e0b/arm{a,b,c}_seed{s}.csv and splits/e0b/e0b_split_summary.csv (sizes and leakage).

Usage (from the repo folder):  python scripts\\make_e0b_splits.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

HAM_CENTERS = [1, 2, 3, 5]
# Table 3 targets T_c, mapped to the Fed-ISIC2019 label codes (MEL 0, NV 1, BCC 2, AK 3, BKL 4, DF 5, VASC 6)
TARGETS = {3: 3027, 2: 3112, 4: 3296, 5: 3342, 0: 3247, 1: 3174, 6: 3322}   # sum 22,520
N_TF = 5            # flips and rotations available for copies (tf 1..5)
N_CLIENTS = 3
ALPHA = 0.5


def balance(df, rng):
    """Undersample classes above target, add flip/rotation copies to classes below target."""
    out = []
    for cls, g in df.groupby("target"):
        t = TARGETS[cls]
        g = g.assign(tf=0)
        if len(g) >= t:
            out.append(g.sample(n=t, replace=False, random_state=rng.integers(1 << 31)))
            continue
        out.append(g)
        need, k, order = t - len(g), 0, []
        while need > 0:                                 # walk through the sources in random order,
            if not order:                               # again and again, until the target is met
                order = list(rng.permutation(len(g)))
                k += 1
            src = g.iloc[order.pop()]
            out.append(src.to_frame().T.assign(tf=(k - 1) % N_TF + 1))
            need -= 1
    res = pd.concat(out, ignore_index=True)
    res["target"] = res["target"].astype(int)
    res["tf"] = res["tf"].astype(int)
    return res


def dirichlet_clients(train, rng):
    """Dirichlet(alpha) class partition of the training rows over N_CLIENTS clients."""
    for _ in range(100):
        client = np.empty(len(train), dtype=int)
        for cls in sorted(train.target.unique()):
            idx = np.flatnonzero(train.target.to_numpy() == cls)
            idx = rng.permutation(idx)
            p = rng.dirichlet([ALPHA] * N_CLIENTS)
            cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
            for c, part in enumerate(np.split(idx, cuts)):
                client[part] = c
        sizes = np.bincount(client, minlength=N_CLIENTS)
        if sizes.min() >= 0.05 * len(train):            # redraw if a client is almost empty
            return client
    raise RuntimeError("could not draw a usable Dirichlet partition")


def make_arm(ham, arm, seed):
    rng = np.random.default_rng(seed)
    if arm == "a":
        g1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        rest, test = next(g1.split(ham, groups=ham.lesion_id))
        rest_df = ham.iloc[rest]
        g2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)   # 0.25 x 0.8 = 0.2
        tr, va = next(g2.split(rest_df, groups=rest_df.lesion_id))
        parts = {"train": rest_df.iloc[tr], "val": rest_df.iloc[va], "test": ham.iloc[test]}
        parts = {k: v.assign(tf=0) for k, v in parts.items()}
        parts["train"] = balance(parts["train"], rng)
    elif arm == "c":
        rest, test = train_test_split(ham, test_size=0.2, stratify=ham.target, random_state=seed)
        tr, va = train_test_split(rest, test_size=0.25, stratify=rest.target, random_state=seed)
        parts = {"train": tr.assign(tf=0), "val": va.assign(tf=0), "test": test.assign(tf=0)}
        parts["train"] = balance(parts["train"], rng)
    elif arm == "b":
        bal = balance(ham, rng)
        rest, test = train_test_split(bal, test_size=0.15, stratify=bal.target, random_state=seed)
        tr, va = train_test_split(rest, test_size=0.15 / 0.85, stratify=rest.target, random_state=seed)
        parts = {"train": tr, "val": va, "test": test}
    else:
        raise ValueError(arm)
    out = []
    for fold, d in parts.items():
        d = d.reset_index(drop=True).assign(fold=fold, client=-1)
        if fold == "train":
            d["client"] = dirichlet_clients(d, rng)
        out.append(d)
    return pd.concat(out, ignore_index=True)[["image", "tf", "target", "lesion_id", "center", "fold", "client"]]


def leakage(df):
    tr, te = df[df.fold == "train"], df[df.fold == "test"]
    return {
        "test_rows_lesion_in_train": round(float(te.lesion_id.isin(set(tr.lesion_id)).mean()), 4),
        "test_rows_source_image_in_train": round(float(te.image.isin(set(tr.image)).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join("splits", "split_official.csv"))
    ap.add_argument("--out", default=os.path.join("splits", "e0b"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()

    df = pd.read_csv(a.split)
    ham = df[df.center.isin(HAM_CENTERS)].reset_index(drop=True)
    print(f"HAM10000 part: {len(ham)} images, {ham.lesion_id.nunique()} lesions, "
          f"classes {dict(ham.target.value_counts().sort_index())}")
    assert len(ham) == 10015 and ham.lesion_id.nunique() == 7470, "unexpected HAM10000 size"
    # ISIC 2019 splits HAM10000's "akiec" class into AK (3) and SCC (7, Bowen's disease).
    # The mother paper uses the 7 HAM10000 classes, so SCC is merged back into akiec (code 3).
    ham["target"] = ham.target.replace({7: 3})
    assert set(ham.target) == set(TARGETS) and (ham.target == 3).sum() == 327, "unexpected HAM10000 labels"

    os.makedirs(a.out, exist_ok=True)
    rows = []
    for seed in a.seeds:
        for arm in "abc":
            s = make_arm(ham, arm, seed)
            s.to_csv(os.path.join(a.out, f"arm{arm}_seed{seed}.csv"), index=False)
            n = s.fold.value_counts()
            r = {"arm": arm, "seed": seed, "train": int(n.get("train", 0)), "val": int(n.get("val", 0)),
                 "test": int(n.get("test", 0)), **leakage(s)}
            r["client_sizes"] = " ".join(str(x) for x in s[s.fold == "train"].client.value_counts().sort_index())
            rows.append(r)
            print(r)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "e0b_split_summary.csv"), index=False)
    print(f"\nsaved {len(rows)} split files to {a.out}")


if __name__ == "__main__":
    main()
