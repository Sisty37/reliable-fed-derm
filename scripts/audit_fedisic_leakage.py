"""
Pilot leakage audit for the official FLamby Fed-ISIC2019 split (version 2).

1. Counts test images whose lesion also appears in the training split
   (overall, per center, per class, cross-center).
2. Optionally uses the ISIC 2019 training metadata to cover all six centers
   (BCN20000 and MSK lesion IDs are only in that file).
3. Writes three training-set variants for a controlled leaky-vs-clean test
   that keeps the official test set unchanged:
     official : original training split (leaky)
     clean    : training images of lesions that appear in test are removed
     control  : the same number of training images removed at random from
                non-leaked lesions (keeps leakage, matches clean's size)

Usage (Windows PowerShell, one line):
  python pilot_fedisic_leakage_v2.py FLamby\\flamby\\datasets\\fed_isic2019
  python pilot_fedisic_leakage_v2.py FLamby\\flamby\\datasets\\fed_isic2019 --isic-meta ISIC_2019_Training_Metadata.csv

Requires: pandas
"""
import argparse
from pathlib import Path

import pandas as pd

LABELS = {0: "MEL", 1: "NV", 2: "BCC", 3: "AK", 4: "BKL", 5: "DF", 6: "VASC", 7: "SCC"}
CENTERS = {0: "BCN20000", 1: "HAM_vidir_molemax", 2: "HAM_vidir_modern",
           3: "HAM_rosendahl", 4: "MSK", 5: "HAM_vienna_dias"}


def load(root: Path, isic_meta: Path | None) -> pd.DataFrame:
    split = pd.read_csv(root / "dataset_creation_scripts" / "train_test_split")
    ham = pd.read_csv(root / "HAM10000_metadata")[["image_id", "lesion_id"]]
    df = split.merge(ham.rename(columns={"image_id": "image"}), on="image", how="left")
    if isic_meta is not None:
        meta = pd.read_csv(isic_meta)[["image", "lesion_id"]].rename(columns={"lesion_id": "lesion_isic"})
        df = df.merge(meta, on="image", how="left")
        df["lesion_id"] = df["lesion_id"].fillna(df["lesion_isic"])
        df = df.drop(columns="lesion_isic")
    df["has_lesion_id"] = df["lesion_id"].notna()
    # images without a lesion ID are treated as their own lesion
    df["lesion_id"] = df["lesion_id"].fillna("IMG_" + df["image"])
    return df


def audit(df: pd.DataFrame) -> pd.DataFrame:
    train_lesions = set(df.loc[df.fold == "train", "lesion_id"])
    test = df[df.fold == "test"].copy()
    test["leaked"] = test.lesion_id.isin(train_lesions)

    known = test[test.has_lesion_id]
    print(f"Images in split: {len(df)} | with lesion ID: {df.has_lesion_id.sum()}")
    print(f"Test images with lesion ID: {len(known)} | lesion also in train (pooled): "
          f"{known.leaked.sum()} ({known.leaked.mean():.1%})")
    print(f"Unique test lesions: {known.lesion_id.nunique()}\n")

    print("Per center (pooled train):")
    for c, g in test.groupby("center"):
        k = g[g.has_lesion_id]
        note = "" if len(k) else "  (no lesion IDs: add --isic-meta)"
        rate = f"{k.leaked.mean():.1%}" if len(k) else "n/a"
        print(f"  {c} {CENTERS.get(c, ''):18s} test {len(g):5d} | with ID {len(k):5d} | "
              f"leaked {int(k.leaked.sum()):5d} ({rate}){note}")

    print("\nPer class (test images with lesion ID):")
    cls = known.assign(cls=known.target.map(LABELS)).groupby("cls").leaked.agg(["sum", "count"])
    cls["rate"] = (cls["sum"] / cls["count"]).round(3)
    print(cls.sort_values("count", ascending=False).to_string())

    span = df[df.has_lesion_id].groupby("lesion_id").center.nunique()
    print(f"\nLesions appearing in more than one center: {(span > 1).sum()}")
    return test


def write_variants(df: pd.DataFrame, out: Path, seed: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    test_lesions = set(df.loc[df.fold == "test", "lesion_id"])
    train = df[df.fold == "train"]
    test = df[df.fold == "test"]

    leaked_train = train[train.lesion_id.isin(test_lesions)]
    clean_train = train.drop(leaked_train.index)

    # control: remove the same number of images per center and class, at random,
    # from training images whose lesion is NOT in the test set
    removed = []
    for (c, t), g in leaked_train.groupby(["center", "target"]):
        pool = clean_train[(clean_train.center == c) & (clean_train.target == t)]
        removed.append(pool.sample(n=min(len(g), len(pool)), random_state=seed))
    control_train = train.drop(pd.concat(removed).index) if removed else train

    cols = ["image", "center", "target", "lesion_id"]
    for name, tr in [("official", train), ("clean", clean_train), ("control", control_train)]:
        v = pd.concat([tr.assign(fold="train"), test.assign(fold="test")])[cols + ["fold"]]
        v.to_csv(out / f"split_{name}.csv", index=False)
        n_leak = tr.lesion_id.isin(test_lesions).sum()
        print(f"  {name:8s}: train {len(tr):5d} | train images sharing a test lesion {n_leak:5d}")
    print(f"Variants written to {out}")


def resplit(df: pd.DataFrame, out: Path, seeds, test_frac: float = 0.2) -> None:
    """Size-matched re-splits from the same pool, per center:
    image-level (FLamby-style, stratified by class) vs lesion-grouped
    (all images of a lesion on one side, stratified by the lesion's label)."""
    out.mkdir(parents=True, exist_ok=True)
    cols = ["image", "center", "target", "lesion_id"]
    print(f"{'seed':>4} {'type':7s} {'train':>6} {'test':>5} {'test leaked':>12}")
    for s in seeds:
        img_parts, les_parts = [], []
        for c, g in df.groupby("center"):
            # image-level split
            test_idx = set(g.groupby("target").sample(frac=test_frac, random_state=s).index)
            img_parts.append(g.assign(fold=["test" if i in test_idx else "train" for i in g.index]))
            # lesion-grouped split: fill test with whole lesions until each class reaches its quota
            les = g.groupby("lesion_id").agg(target=("target", lambda t: t.mode().iat[0]),
                                             n=("image", "size")).sample(frac=1, random_state=s)
            quota = (g.target.value_counts() * test_frac).round().astype(int).to_dict()
            filled = {k: 0 for k in quota}
            test_les = []
            for lid, row in les.iterrows():
                t = row.target
                if filled.get(t, 0) + row.n <= quota.get(t, 0) or filled.get(t, 0) == 0 and quota.get(t, 0) > 0:
                    test_les.append(lid)
                    filled[t] = filled.get(t, 0) + row.n
            les_parts.append(g.assign(fold=g.lesion_id.isin(set(test_les)).map({True: "test", False: "train"})))
        for name, parts in [("image", img_parts), ("lesion", les_parts)]:
            v = pd.concat(parts)
            if name == "lesion":
                # lesions that span two centers: keep all their images in train
                tr_les = set(v.loc[v.fold == "train", "lesion_id"])
                v.loc[(v.fold == "test") & v.lesion_id.isin(tr_les), "fold"] = "train"
            tr_les = set(v.loc[v.fold == "train", "lesion_id"])
            te = v[v.fold == "test"]
            leak = te.lesion_id.isin(tr_les).mean()
            print(f"{s:>4} {name:7s} {(v.fold == 'train').sum():>6} {len(te):>5} {leak:>11.1%}")
            v[cols + ["fold"]].to_csv(out / f"resplit_{name}_seed{s}.csv", index=False)
    print(f"Re-splits written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="FLamby/flamby/datasets/fed_isic2019")
    ap.add_argument("--isic-meta", type=Path, default=None, help="ISIC_2019_Training_Metadata.csv")
    ap.add_argument("--out", type=Path, default=Path("results/splits"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resplit-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()

    data = load(a.root, a.isic_meta)
    audit(data)
    print("\nWriting training-set variants (official test set kept unchanged):")
    write_variants(data, a.out, a.seed)
    print("\nSize-matched re-splits (image-level vs lesion-grouped, 20% test per center):")
    resplit(data, a.out, a.resplit_seeds)
