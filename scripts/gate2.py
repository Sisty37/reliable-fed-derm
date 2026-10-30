"""
Gate 2 (protocol v0.10, Sections 3, 9 and 10): leakage inflation on the image-level re-splits.

Primary outcome: model A FedAvg, 9 rounds, seeds 0-2.
  inflation = BA on its own image-level test set  -  reweighted BA on the new-lesion subset of that
              test set (test images whose lesion is not in training), same model.
  Reweighting: for each class, recall on the new-lesion subset is the weighted mean of per-center
  recalls, weights = each center's share of that class in the full test set, renormalized over the
  centers that have new-lesion images of that class. Classes with < 5 new-lesion images (pooled over
  centers) are dropped from both terms.
  Interval: lesion-wise cluster bootstrap (2,000 resamples of test lesions); the same resampled
  lesions are used for both terms; seeds are averaged inside each replicate.
  Gate 2 passes if inflation >= 2 points and the 95% interval is above zero.

Secondary (reported, no decision): the same inflation for FedAvg at 27 rounds and for pooled training;
the between-condition difference (BA of the image-level model on its own test set minus BA of the
lesion-grouped model on its own test set); melanoma recall in both terms.

Usage (from the repo folder):  python scripts\\gate2.py
Writes results/E1_gate2.json and prints a summary.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

MIN_NEW = 5
MEL = 0


def load_run(run_dir, split_csv):
    p = np.load(os.path.join(run_dir, "preds.npz"))
    df = pd.DataFrame({"image": p["image"].astype(str), "center": p["center"], "target": p["target"],
                       "pred": p["logits"].argmax(1)})
    sp = pd.read_csv(split_csv)
    df = df.merge(sp[sp.fold == "test"][["image", "lesion_id"]], on="image", how="left")
    assert df.lesion_id.notna().all(), f"test images missing from {split_csv}"
    train_les = set(sp.loc[sp.fold == "train", "lesion_id"])
    df["new"] = ~df.lesion_id.isin(train_les)
    df["correct"] = (df.pred == df.target).astype(float)
    return df


def kept_classes(df):
    n_new = df[df.new].groupby("target").size()
    return sorted(n_new[n_new >= MIN_NEW].index)


def terms(df, w, classes):
    """BA on the full test set and reweighted BA on the new-lesion subset, with image weights w."""
    full, new = [], []
    for k in classes:
        m = (df.target == k).to_numpy()
        wk = w[m]
        if wk.sum() == 0:
            continue
        full_rec = np.sum(wk * df.correct.to_numpy()[m]) / wk.sum()
        sub = df[m]
        cen = sub.center.to_numpy()
        isnew = sub.new.to_numpy()
        cor = sub.correct.to_numpy()
        num = den = 0.0
        for c in np.unique(cen):
            mc = cen == c
            share = wk[mc].sum()                                 # center's share of class k (full test)
            wn = wk[mc & isnew]
            if share == 0 or wn.sum() == 0:
                continue                                         # renormalize over centers with new images
            num += share * np.sum(wn * cor[mc & isnew]) / wn.sum()
            den += share
        if den == 0:
            continue                                             # class has no new images in this replicate
        full.append(full_rec)
        new.append(num / den)
    return float(np.mean(full)), float(np.mean(new)), (full, new)


def mel_terms(df, w):
    m = (df.target == MEL).to_numpy()
    full = np.sum(w[m] * df.correct.to_numpy()[m]) / w[m].sum()
    mn = m & df.new.to_numpy()
    new = np.sum(w[mn] * df.correct.to_numpy()[mn]) / w[mn].sum() if w[mn].sum() else np.nan
    return full, new


def lesion_weights(df, rng):
    """Multiplicity of each image's lesion in one cluster bootstrap resample of the test lesions."""
    les, inv = np.unique(df.lesion_id.to_numpy(), return_inverse=True)
    counts = rng.multinomial(len(les), np.full(len(les), 1 / len(les)))
    return counts[inv].astype(float)


def inflation(runs, n_boot, seed):
    """runs: list of (seed, df). Returns point estimate, interval and per-seed values."""
    per_seed = []
    for s, df in runs:
        cls = kept_classes(df)
        f, n, _ = terms(df, np.ones(len(df)), cls)
        mf, mn = mel_terms(df, np.ones(len(df)))
        per_seed.append({"seed": s, "classes_kept": cls, "ba_full": f, "ba_new_reweighted": n,
                         "inflation": f - n, "n_test": len(df), "n_new": int(df.new.sum()),
                         "mel_recall_full": mf, "mel_recall_new": mn})
    point = float(np.mean([r["inflation"] for r in per_seed]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        vals = []
        for (s, df), r in zip(runs, per_seed):
            f, n, _ = terms(df, lesion_weights(df, rng), r["classes_kept"])
            vals.append(f - n)
        boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"inflation": point, "ci95": [float(lo), float(hi)], "per_seed": per_seed,
            "seed_range": [min(r["inflation"] for r in per_seed), max(r["inflation"] for r in per_seed)]}


def plain_ba(df):
    return float(np.mean([df.correct[df.target == k].mean() for k in sorted(df.target.unique())]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split-dir", default=os.path.join("splits", "e1"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join("results", "E1_gate2.json"))
    a = ap.parse_args()

    def runs_for(kind, tag):
        out = []
        for s in a.seeds:
            name = f"resplit_{kind}_seed{s}_val"
            d = os.path.join(a.runs, name, tag.format(s=s))
            if os.path.exists(os.path.join(d, "preds.npz")):
                out.append((s, load_run(d, os.path.join(a.split_dir, name + ".csv"))))
        return out

    res = {}
    for label, tag in [("fedavg_r9 (PRIMARY)", "fedavg_r9_s{s}"), ("fedavg_r27", "fedavg_r27_s{s}"),
                       ("pooled_e20", "pooled_e20_s{s}")]:
        img = runs_for("image", tag)
        if len(img) < len(a.seeds):
            print(f"{label}: only {len(img)} of {len(a.seeds)} image-level runs found, skipped")
            continue
        r = inflation(img, a.n_boot, seed=12345)
        les = runs_for("lesion", tag)
        if len(les) == len(a.seeds):
            ba_img = np.mean([plain_ba(df) for _, df in img])
            ba_les = np.mean([plain_ba(df) for _, df in les])
            r["between_condition"] = {"ba_image_model_own_test": float(ba_img),
                                      "ba_lesion_model_own_test": float(ba_les),
                                      "difference": float(ba_img - ba_les)}
        res[label] = r
        print(f"\n{label}")
        for p in r["per_seed"]:
            print(f"  seed {p['seed']}: BA full {p['ba_full']:.4f} | BA new (reweighted) {p['ba_new_reweighted']:.4f}"
                  f" | inflation {100 * p['inflation']:+.2f} pts | new-lesion images {p['n_new']}/{p['n_test']}"
                  f" | melanoma recall {p['mel_recall_full']:.3f} vs {p['mel_recall_new']:.3f}")
        print(f"  inflation {100 * r['inflation']:+.2f} pts, 95% CI [{100 * r['ci95'][0]:+.2f}, "
              f"{100 * r['ci95'][1]:+.2f}], seed range [{100 * r['seed_range'][0]:+.2f}, {100 * r['seed_range'][1]:+.2f}]")
        if "between_condition" in r:
            b = r["between_condition"]
            print(f"  between-condition (sensitivity): {b['ba_image_model_own_test']:.4f} - "
                  f"{b['ba_lesion_model_own_test']:.4f} = {100 * b['difference']:+.2f} pts")

    if "fedavg_r9 (PRIMARY)" in res:
        r = res["fedavg_r9 (PRIMARY)"]
        passed = r["inflation"] >= 0.02 and r["ci95"][0] > 0
        res["gate2"] = "PASS: RQ2 is the headline" if passed else "NOT MET: RQ1 is the headline, RQ2 reported as small/null"
        print(f"\nGATE 2: inflation {100 * r['inflation']:+.2f} pts, CI lower {100 * r['ci95'][0]:+.2f}  ->  {res['gate2']}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
