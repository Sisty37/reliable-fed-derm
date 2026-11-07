"""
Exploratory nearest-neighbour proximity, matched by client and class (review point 99; post hoc).

Extends scripts/nn_proximity.py. For each frozen encoder and image-level split (seeds 0-4), with centred,
L2-normalized plain-view embeddings (training mean of that split):
  1. cosine similarity of each test image to its nearest training image, leaked vs new-lesion images:
     raw difference, and the client x class matched difference (mean of within-stratum differences,
     weighted by the stratum's new-lesion count; strata need >= 3 leaked and >= 3 new images);
  2. the same for the nearest training image from ANOTHER lesion (siblings excluded): if leaked images are
     no closer to other lesions than new images are, their excess proximity comes from their siblings
     rather than from lying in denser regions of the feature space;
  3. 1-nearest-neighbour balanced accuracy on leaked vs new images, standardized like Eq. (1)
     (per-class recalls of the new subset reweighted by each client's share of the class), and the same with
     siblings removed from the training set (1-NN without siblings).

Usage (from the repo folder):  python scripts\\nn_matched.py
Writes results/NN_matched.json and prints the tables. Needs embeddings/effb0.npz and embeddings/panderm.npz.
"""
import json
import os

import numpy as np
import pandas as pd

ENCODERS = {"effb0": "A-frozen", "panderm": "B"}
K, C, MIN_N = 8, 6, 3


def nearest(test, train, chunk=512, exclude=None):
    """Index and similarity of the nearest training row; exclude[i] = training mask to ignore for test row i."""
    idx, sim = np.empty(len(test), int), np.empty(len(test), np.float32)
    for i in range(0, len(test), chunk):
        s = test[i:i + chunk] @ train.T
        if exclude is not None:
            s = np.where(exclude[i:i + chunk], -np.inf, s)
        idx[i:i + chunk] = s.argmax(1)
        sim[i:i + chunk] = s.max(1)
    return idx, sim


def matched_diff(val, leaked, cell):
    diffs, wts = [], []
    for k in np.unique(cell):
        m = cell == k
        a, b = val[m & leaked], val[m & ~leaked]
        if len(a) >= MIN_N and len(b) >= MIN_N:
            diffs.append(a.mean() - b.mean())
            wts.append(len(b))
    return float(np.average(diffs, weights=wts)) if diffs else np.nan


def std_ba(pred, y, center, sub):
    """Eq. (1)-style: per-class recall over clients, weighted by each client's share of the class in the full set."""
    rec = []
    for k in range(K):
        num = den = 0.0
        for c in range(C):
            full = (y == k) & (center == c)
            s = full & sub
            if s.sum() == 0:
                continue
            num += full.sum() * np.mean(pred[s] == k)
            den += full.sum()
        if den > 0 and ((y == k) & sub).sum() >= 5:
            rec.append(num / den)
    return float(np.mean(rec))


def main():
    res = {}
    for enc, label in ENCODERS.items():
        z = np.load(os.path.join("embeddings", f"{enc}.npz"))
        names, plain = z["image"].astype(str), z["plain"].astype(np.float32)
        pos = {n: i for i, n in enumerate(names)}
        rows = []
        for s in range(5):
            sp = pd.read_csv(os.path.join("splits", "e1", f"resplit_image_seed{s}_val.csv"))
            tr, te = sp[sp.fold == "train"], sp[sp.fold == "test"]
            itr, ite = tr.image.astype(str).map(pos).to_numpy(), te.image.astype(str).map(pos).to_numpy()
            X = plain - plain[itr].mean(0)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            Xtr, Xte = X[itr], X[ite]
            y, ytr = te.target.to_numpy(), tr.target.to_numpy()
            les, les_tr = te.lesion_id.to_numpy(), tr.lesion_id.to_numpy()
            center = te.center.to_numpy()
            leaked = np.isin(les, list(set(les_tr)))
            cell = center * K + y
            nn_i, nn_s = nearest(Xte, Xtr)
            lt_codes = pd.factorize(np.concatenate([les_tr, les]))[0]
            ctr, cte = lt_codes[:len(les_tr)], lt_codes[len(les_tr):]
            excl = cte[:, None] == ctr[None, :]                       # siblings of each test image
            nn2_i, nn2_s = nearest(Xte, Xtr, exclude=excl)
            pred1, pred2 = ytr[nn_i], ytr[nn2_i]
            full = np.ones(len(y), bool)
            r = {"seed": s,
                 "sim_raw_diff": float(nn_s[leaked].mean() - nn_s[~leaked].mean()),
                 "sim_matched_diff": matched_diff(nn_s, leaked, cell),
                 "sim_other_lesion_raw_diff": float(nn2_s[leaked].mean() - nn2_s[~leaked].mean()),
                 "sim_other_lesion_matched_diff": matched_diff(nn2_s, leaked, cell),
                 "nn1_std_inflation": std_ba(pred1, y, center, full) - std_ba(pred1, y, center, ~leaked),
                 "nn1_no_sibling_std_inflation": std_ba(pred2, y, center, full) - std_ba(pred2, y, center, ~leaked)}
            rows.append(r)
        df = pd.DataFrame(rows)
        res[enc] = {"per_seed": rows, "mean": df.mean(numeric_only=True).to_dict(), "sd": df.std(numeric_only=True).to_dict()}
        m, sd = res[enc]["mean"], res[enc]["sd"]
        print(f"\n=== {enc} ({label}); image-level splits, seeds {df.seed.tolist()}; mean (sd) over seeds ===")
        print(f"  nearest-train cosine, leaked - new        raw {m['sim_raw_diff']:+.3f} ({sd['sim_raw_diff']:.3f}) | "
              f"client x class matched {m['sim_matched_diff']:+.3f} ({sd['sim_matched_diff']:.3f})")
        print(f"  nearest OTHER-lesion cosine, leaked - new raw {m['sim_other_lesion_raw_diff']:+.3f} | "
              f"matched {m['sim_other_lesion_matched_diff']:+.3f} ({sd['sim_other_lesion_matched_diff']:.3f})")
        print(f"  1-NN standardized inflation (pts)          with siblings {100 * m['nn1_std_inflation']:+.1f} "
              f"({100 * sd['nn1_std_inflation']:.1f}) | siblings removed {100 * m['nn1_no_sibling_std_inflation']:+.1f} "
              f"({100 * sd['nn1_no_sibling_std_inflation']:.1f})")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(os.path.join("results", "NN_matched.json"), "w"), indent=2, default=float)
    print("\nsaved results/NN_matched.json")


if __name__ == "__main__":
    main()
