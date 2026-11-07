"""
Post hoc sensitivity analyses of the calibration results (protocol v0.12, Section 12, 7 Nov 2026).
No new training: E2 predictions (FedAvg and pooled, image-level and lesion-grouped, seeds 0-4), models A and B.

  1. Size- and composition-matched ECE: on each image-level test set, ECE of the new-lesion subset minus the
     mean ECE of 200 random subsets of the full test set with the same size and the same center x class
     counts as the new-lesion subset (removes the finite-sample bias of binned ECE and the class-mix
     difference). The unmatched contrast of E3 is shown next to it.
  2. Bin-free metrics: NLL and Brier score of the new-lesion subset minus those of the full set reweighted
     to the new subset's center x class composition.
  3. ECE binning: 10, 15, 20 equal-width bins and 15 equal-mass bins (unmatched and matched contrast).
  4. Global temperature: NLL-optimal temperature on the pooled validation set (all centers) instead of the
     image-weighted mean of per-center temperatures; ECE of every model both ways.
  5. Validation exposure (image-level splits): share of test images whose lesion is in validation but not in
     training; contrasts 1-2 with those images removed from the new-lesion subset.
Intervals: 500 lesion-bootstrap replicates, seeds averaged inside each replicate.

Usage (from the repo folder):  python scripts/calibration_sensitivity.py
Output: results/Calibration_sensitivity.json (+ printed tables).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5_conformal import fit_temperature, softmax  # noqa: E402

N_C, N_K = 6, 8
SEEDS = [0, 1, 2, 3, 4]


def ece(p, y, w=None, bins=15, mass=False):
    w = np.ones(len(y)) if w is None else w
    keep = w > 0
    p, y, w = p[keep], y[keep], w[keep]
    conf = p.max(1)
    acc = (p.argmax(1) == y).astype(float)
    if mass:
        order = np.argsort(conf, kind="stable")
        cw = np.cumsum(w[order]) / w.sum()
        b = np.empty(len(y), int)
        b[order] = np.minimum((cw * bins - 1e-9).astype(int), bins - 1)
    else:
        b = np.minimum((conf * bins).astype(int), bins - 1)
    out = 0.0
    for i in range(bins):
        m = b == i
        if w[m].sum() > 0:
            out += w[m].sum() / w.sum() * abs(np.average(acc[m], weights=w[m]) - np.average(conf[m], weights=w[m]))
    return out


def nll(p, y, w):
    return float(np.sum(w * -np.log(p[np.arange(len(y)), y] + 1e-12)) / w.sum())


def brier(p, y, w):
    return float(np.sum(w * ((p - np.eye(N_K)[y]) ** 2).sum(1)) / w.sum())


def load(runs, split_dir, kind, seed, run, temp_rule):
    name = f"resplit_{kind}_seed{seed}_val"
    sp = pd.read_csv(os.path.join(split_dir, name + ".csv")).set_index("image")
    train_les = set(sp.loc[sp.fold == "train", "lesion_id"])
    val_les = set(sp.loc[sp.fold == "val", "lesion_id"])
    d = os.path.join(runs, name, run)
    out = {}
    for fold, f in (("test", "preds.npz"), ("val", "preds_val.npz")):
        z = np.load(os.path.join(d, f))
        les = sp.loc[z["image"].astype(str), "lesion_id"].to_numpy()
        out[fold] = {"logits": z["logits"].astype(np.float64), "y": z["target"].astype(int),
                     "center": z["center"].astype(int), "lesion": les,
                     "new": ~pd.Series(les).isin(train_les).to_numpy(),
                     "val_only": (~pd.Series(les).isin(train_les) & pd.Series(les).isin(val_les)).to_numpy()}
    v = out["val"]
    if temp_rule == "mean":
        temps, sizes = [], []
        for c in range(N_C):
            m = v["center"] == c
            if m.sum():
                temps.append(fit_temperature(v["logits"][m], v["y"][m]))
                sizes.append(int(m.sum()))
        t = float(np.average(temps, weights=sizes))
    else:
        t = fit_temperature(v["logits"], v["y"])
    o = out["test"]
    o["p"] = softmax(o["logits"] / t)
    o["T"] = t
    o["cell"] = o["center"] * N_K + o["y"]
    return o


def lesion_weights(o, rng):
    u, inv = np.unique(o["lesion"], return_inverse=True)
    return rng.multinomial(len(u), np.full(len(u), 1 / len(u)))[inv].astype(float)


def matched_weights(o, sub, w):
    """Weights on all test images that give the full set the center x class composition of subset `sub`."""
    target = np.bincount(o["cell"][sub], weights=w[sub], minlength=N_C * N_K)
    have = np.bincount(o["cell"], weights=w, minlength=N_C * N_K)
    ratio = np.where(have > 0, target / np.maximum(have, 1e-12), 0.0)
    return w * ratio[o["cell"]]


def matched_subsample_ece(o, sub, w, rng, reps, **kw):
    """Mean ECE over random subsets of the full test set with the subset's size and cell counts."""
    pos = np.flatnonzero(w > 0)
    need = np.bincount(o["cell"][sub], weights=w[sub], minlength=N_C * N_K).round().astype(int)
    by_cell = [pos[o["cell"][pos] == k] for k in range(N_C * N_K)]
    vals = []
    for _ in range(reps):
        parts = []
        for k in range(N_C * N_K):
            if need[k] == 0 or not len(by_cell[k]):
                continue
            wk = w[by_cell[k]]
            plain = np.all(wk == 1) and need[k] <= len(by_cell[k])    # without replacement when possible
            parts.append(rng.choice(by_cell[k], size=need[k], replace=not plain, p=None if plain else wk / wk.sum()))
        idx = np.concatenate(parts)
        vals.append(ece(o["p"][idx], o["y"][idx], **kw))
    return float(np.mean(vals))


def contrasts(os_, ws, rng, reps, exclude_val=False):
    out = {k: [] for k in ("ece_unmatched", "ece_matched", "nll", "brier", "ece10", "ece20", "ece_mass15",
                           "ece_mass15_matched")}
    for o, w in zip(os_, ws):
        sub = o["new"] & (~o["val_only"] if exclude_val else True)
        wm = matched_weights(o, sub, w)
        out["ece_unmatched"].append(ece(o["p"][sub], o["y"][sub], w[sub]) - ece(o["p"], o["y"], w))
        out["ece_matched"].append(ece(o["p"][sub], o["y"][sub], w[sub]) - matched_subsample_ece(o, sub, w, rng, reps))
        out["nll"].append(nll(o["p"][sub], o["y"][sub], w[sub]) - nll(o["p"], o["y"], wm))
        out["brier"].append(brier(o["p"][sub], o["y"][sub], w[sub]) - brier(o["p"], o["y"], wm))
        out["ece10"].append(ece(o["p"][sub], o["y"][sub], w[sub], bins=10) - ece(o["p"], o["y"], w, bins=10))
        out["ece20"].append(ece(o["p"][sub], o["y"][sub], w[sub], bins=20) - ece(o["p"], o["y"], w, bins=20))
        out["ece_mass15"].append(ece(o["p"][sub], o["y"][sub], w[sub], mass=True) - ece(o["p"], o["y"], w, mass=True))
        out["ece_mass15_matched"].append(ece(o["p"][sub], o["y"][sub], w[sub], mass=True)
                                         - matched_subsample_ece(o, sub, w, rng, reps, mass=True))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split-dir", default=os.path.join("splits", "e1"))
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default=os.path.join("results", "Calibration_sensitivity.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(20261107)
    res = {}
    for model, prefix in (("A", ""), ("B", "pandermfrozen_")):
        for m in ("fedavg_r9", "pooled_e20"):
            key = f"{model}/{m}"
            img = [load(a.runs, a.split_dir, "image", s, f"{prefix}{m}_s{s}", "mean") for s in SEEDS]
            r = {"val_only_share_of_test": float(np.mean([o["val_only"].mean() for o in img])),
                 "val_only_share_of_new": float(np.mean([o["val_only"][o["new"]].mean() for o in img]))}
            for excl in (False, True):
                ones = [np.ones(len(o["y"])) for o in img]
                pt = contrasts(img, ones, rng, a.reps, excl)
                bs = [contrasts(img, [lesion_weights(o, rng) for o in img], rng, max(a.reps // 10, 10), excl)
                      for _ in range(a.n_boot)]
                r["excluding_val_exposed" if excl else "main"] = {
                    k: [pt[k], float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))]
                    for k in pt}
            # temperature rule
            tr = {}
            for kind in ("image", "lesion"):
                for rule in ("mean", "pooled_nll"):
                    os_ = [load(a.runs, a.split_dir, kind, s, f"{prefix}{m}_s{s}", rule) for s in SEEDS]
                    tr[f"{kind}|{rule}"] = {"T": float(np.mean([o["T"] for o in os_])),
                                            "ece": float(np.mean([ece(o["p"], o["y"]) for o in os_])),
                                            "nll": float(np.mean([nll(o["p"], o["y"], np.ones(len(o["y"]))) for o in os_]))}
            r["temperature_rule"] = tr
            res[key] = r
            print(f"\n=== model {model}, {m} (image-level test sets, 5 seeds; points x100 for ECE/Brier) ===")
            print(f"  test images whose lesion is in validation only: {100 * r['val_only_share_of_test']:.1f}% "
                  f"of test, {100 * r['val_only_share_of_new']:.1f}% of new-lesion images")
            for part in ("main", "excluding_val_exposed"):
                print(f"  [{part}]")
                for k, (p, lo, hi) in r[part].items():
                    f = 1 if k == "nll" else 100
                    print(f"    new minus full, {k:20s} {f * p:+7.2f} [{f * lo:+.2f}, {f * hi:+.2f}]")
            for k, v in tr.items():
                print(f"  temperature {k:20s} T = {v['T']:.3f}  ECE {100 * v['ece']:.2f}  NLL {v['nll']:.3f}")
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
