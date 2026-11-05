"""
Exploratory: does "proximity" explain the leakage inflation of frozen-encoder heads? (labelled
exploratory, not in the protocol; choices fixed in results/NN_proximity_notes.md before running).

For each frozen encoder (effb0 = model A-frozen, panderm = model B) and each image-level split with a
FedAvg head (seeds found on disk), using the plain-view embeddings, centred with the training mean and
L2-normalized (cosine similarity):
  1. nearest training image of every test image; its similarity, and whether it is from the same lesion;
     compared between leaked test images (lesion in training) and new-lesion test images;
  2. a 1-nearest-neighbour classifier (label of the nearest training image): balanced accuracy on the
     leaked and the new-lesion test images, and its inflation (full test set minus new-lesion subset);
  3. the FedAvg linear head: accuracy on leaked images whose nearest neighbour is a sibling image of the
     same lesion vs leaked images whose nearest neighbour is not, and on new-lesion images; and how often
     the head agrees with the 1-NN label.
Balanced accuracy here is unweighted over the classes present (no center reweighting), so it is
descriptive and not comparable number-for-number with the Gate 2 measure.

Usage (from the repo folder):  python scripts\\nn_proximity.py
Writes results/NN_proximity.json and prints the tables.
"""
import json
import os

import numpy as np
import pandas as pd

ENCODERS = {"effb0": "effb0frozen_", "panderm": "pandermfrozen_"}


def bacc(pred, y):
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))


def nearest(test, train, chunk=1024):
    idx, sim = np.empty(len(test), int), np.empty(len(test), np.float32)
    for i in range(0, len(test), chunk):
        s = test[i:i + chunk] @ train.T
        idx[i:i + chunk] = s.argmax(1)
        sim[i:i + chunk] = s.max(1)
    return idx, sim


def main():
    res = {}
    for enc, prefix in ENCODERS.items():
        z = np.load(os.path.join("embeddings", f"{enc}.npz"))
        names, plain = z["image"].astype(str), z["plain"].astype(np.float32)
        pos = {n: i for i, n in enumerate(names)}
        rows = []
        for s in range(5):
            run = os.path.join("runs", f"resplit_image_seed{s}_val", f"{prefix}fedavg_r9_s{s}", "preds.npz")
            if not os.path.exists(run):
                continue
            sp = pd.read_csv(os.path.join("splits", "e1", f"resplit_image_seed{s}_val.csv"))
            tr, te = sp[sp.fold == "train"], sp[sp.fold == "test"]
            X = plain - plain[tr.image.astype(str).map(pos).to_numpy()].mean(0)
            X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            Xtr, Xte = X[tr.image.astype(str).map(pos).to_numpy()], X[te.image.astype(str).map(pos).to_numpy()]
            nn_i, nn_s = nearest(Xte, Xtr)
            y, ytr = te.target.to_numpy(), tr.target.to_numpy()
            les, les_tr = te.lesion_id.to_numpy(), tr.lesion_id.to_numpy()
            leaked = np.isin(les, list(set(les_tr)))
            sib = les_tr[nn_i] == les
            nn_pred = ytr[nn_i]
            p = np.load(run)
            head = pd.Series(p["logits"].argmax(1), index=p["image"].astype(str)).loc[te.image.astype(str)].to_numpy()
            new = ~leaked
            r = {"seed": s, "n_test": len(te), "leaked_share": float(leaked.mean()),
                 "sim_leaked": float(nn_s[leaked].mean()), "sim_new": float(nn_s[new].mean()),
                 "nn_is_sibling_leaked": float(sib[leaked].mean()),
                 "nn1_ba_full": bacc(nn_pred, y), "nn1_ba_leaked": bacc(nn_pred[leaked], y[leaked]),
                 "nn1_ba_new": bacc(nn_pred[new], y[new]),
                 "head_ba_full": bacc(head, y), "head_ba_new": bacc(head[new], y[new]),
                 "head_acc_leaked_sibling_nn": float(np.mean(head[leaked & sib] == y[leaked & sib])),
                 "head_acc_leaked_other_nn": float(np.mean(head[leaked & ~sib] == y[leaked & ~sib])) if (leaked & ~sib).any() else np.nan,
                 "head_acc_new": float(np.mean(head[new] == y[new])),
                 "head_agrees_nn_leaked": float(np.mean(head[leaked] == nn_pred[leaked])),
                 "head_agrees_nn_new": float(np.mean(head[new] == nn_pred[new]))}
            r["nn1_inflation"] = r["nn1_ba_full"] - r["nn1_ba_new"]
            r["head_inflation"] = r["head_ba_full"] - r["head_ba_new"]
            rows.append(r)
        df = pd.DataFrame(rows)
        res[enc] = {"per_seed": rows, "mean": df.mean(numeric_only=True).to_dict(), "sd": df.std(numeric_only=True).to_dict()}
        m, sd = res[enc]["mean"], res[enc]["sd"]
        print(f"\n=== {enc} (frozen; image-level splits, seeds {df.seed.tolist()}; mean, sd over seeds) ===")
        print(f"  leaked share of test images            {m['leaked_share']:.3f}")
        print(f"  nearest-train cosine similarity        leaked {m['sim_leaked']:.3f} | new {m['sim_new']:.3f}")
        print(f"  nearest neighbour is a sibling image   {m['nn_is_sibling_leaked']:.3f} of leaked test images")
        print(f"  1-NN balanced accuracy                 full {m['nn1_ba_full']:.3f} | leaked {m['nn1_ba_leaked']:.3f} | new {m['nn1_ba_new']:.3f}"
              f" | inflation {100 * m['nn1_inflation']:+.1f} pts (sd {100 * sd['nn1_inflation']:.1f})")
        print(f"  linear head (FedAvg) balanced accuracy full {m['head_ba_full']:.3f} | new {m['head_ba_new']:.3f}"
              f" | inflation {100 * m['head_inflation']:+.1f} pts (sd {100 * sd['head_inflation']:.1f})")
        print(f"  head accuracy: leaked, sibling NN {m['head_acc_leaked_sibling_nn']:.3f} | leaked, other NN "
              f"{m['head_acc_leaked_other_nn']:.3f} | new {m['head_acc_new']:.3f}")
        print(f"  head agrees with 1-NN label            leaked {m['head_agrees_nn_leaked']:.3f} | new {m['head_agrees_nn_new']:.3f}")
    json.dump(res, open(os.path.join("results", "NN_proximity.json"), "w"), indent=2, default=float)
    print("\nsaved results/NN_proximity.json")


if __name__ == "__main__":
    main()
