"""
E3 (calibration, protocol v0.11 Sections 5 and 7) and E6 (unseen centers, RQ1), from saved predictions.

Part 1, calibration of the E2 models (model A: 7 methods; model B: 6), image-level vs lesion-grouped,
seeds 0-4. Probabilities with the global temperature (each center's validation lesions, NLL; mean
weighted by validation images; personalized methods use the same rule on their own predictions).
Metrics: top-label ECE (15 equal-width bins), class-wise ECE (mean over classes present of the
one-vs-rest ECE, 15 bins), Brier score; raw (T = 1) ECE as well.
  Leakage contrasts (FedAvg and pooled), lesion bootstrap (1,000 replicates, seeds averaged inside):
    within-model: ECE on the new-lesion subset of the image-level test set minus ECE on the full set;
    between-condition: ECE of the image-level model on its test set minus that of the lesion-grouped
    model on its own test set.
Part 2, conformal coverage under leakage (FedAvg, LAC, alpha 0.1): on the image-level test sets,
  thresholds are calibrated on a random lesion half (global and client-wise; 100 splits) and coverage
  is measured separately on evaluation images whose lesion is in training ("leaked") and on the rest
  ("new"); the lesion-grouped models give the clean reference.
Part 3, E6 unseen centers: for each held-out center h and seed s, balanced accuracy (classes present)
  and ECE of the leave-one-center-out model on center h, against the federation model that trained on
  center h (lesion-grouped FedAvg, same seed).

Usage (from the repo folder):
  python scripts\\e3_e6_analysis.py                          # model A
  python scripts\\e3_e6_analysis.py --prefix pandermfrozen_   # model B
Writes results/E3_E6_analysis[_<prefix>].json and prints the tables.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5_conformal import fit_temperature, softmax  # noqa: E402

N_C, N_K = 6, 8
METHODS_A = ["fedavg_r9", "fedprox_r9", "scaffold_r9", "fedbn_r9", "fedavgft_r9", "pooled_e20", "local_e20"]


def ece(p, y, w=None, bins=15):
    w = np.ones(len(y)) if w is None else w
    conf = p.max(1)
    acc = (p.argmax(1) == y).astype(float)
    b = np.minimum((conf * bins).astype(int), bins - 1)
    tot = w.sum()
    out = 0.0
    for i in range(bins):
        m = b == i
        if w[m].sum() > 0:
            out += w[m].sum() / tot * abs(np.average(acc[m], weights=w[m]) - np.average(conf[m], weights=w[m]))
    return out


def classwise_ece(p, y, bins=15):
    vals = []
    for k in np.unique(y):
        pk = p[:, k]
        yk = (y == k).astype(float)
        b = np.minimum((pk * bins).astype(int), bins - 1)
        e = sum(np.mean(b == i) * abs(yk[b == i].mean() - pk[b == i].mean()) for i in range(bins) if (b == i).any())
        vals.append(e)
    return float(np.mean(vals))


def brier(p, y):
    oh = np.eye(N_K)[y]
    return float(np.mean(((p - oh) ** 2).sum(1)))


def bacc(p, y):
    pred = p.argmax(1)
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))


def load(runs, split_dir, kind, seed, run):
    name = f"resplit_{kind}_seed{seed}_val"
    sp = pd.read_csv(os.path.join(split_dir, name + ".csv")).set_index("image")
    train_les = set(sp.loc[sp.fold == "train", "lesion_id"])
    d = os.path.join(runs, name, run)
    out = {}
    for fold, f in (("test", "preds.npz"), ("val", "preds_val.npz")):
        z = np.load(os.path.join(d, f))
        les = sp.loc[z["image"].astype(str), "lesion_id"].to_numpy()
        out[fold] = {"logits": z["logits"].astype(np.float64), "y": z["target"].astype(int),
                     "center": z["center"].astype(int), "lesion": les, "new": ~pd.Series(les).isin(train_les).to_numpy()}
    v = out["val"]
    temps, sizes = [], []
    for c in range(N_C):
        m = v["center"] == c
        if m.sum():
            temps.append(fit_temperature(v["logits"][m], v["y"][m]))
            sizes.append(int(m.sum()))
    t = float(np.average(temps, weights=sizes))
    o = out["test"]
    o["p_raw"] = softmax(o["logits"])
    o["p"] = softmax(o["logits"] / t)
    o["T"] = t
    return o


def lesion_weights(o, rng):
    u, inv = np.unique(o["lesion"], return_inverse=True)
    return rng.multinomial(len(u), np.full(len(u), 1 / len(u)))[inv].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split-dir", default=os.path.join("splits", "e1"))
    ap.add_argument("--prefix", default="")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    methods = [m for m in METHODS_A if not (a.prefix and m == "fedbn_r9")]
    out_path = a.out or os.path.join("results", f"E3_E6_analysis{'_' + a.prefix.rstrip('_') if a.prefix else ''}.json")
    rng = np.random.default_rng(20261104)
    seeds = [0, 1, 2, 3, 4]
    res = {"calibration": {}, "leakage_contrasts": {}, "conformal_leakage": {}, "e6": {}}

    # ---------------- Part 1: calibration of E2 models
    data = {}
    for m in methods:
        for k in ("image", "lesion"):
            for s in seeds:
                data[(m, k, s)] = load(a.runs, a.split_dir, k, s, f"{a.prefix}{m}_s{s}")
    print(f"\nModel {'B' if a.prefix else 'A'}: calibration on the test sets (temperature-scaled; mean of 5 seeds)")
    print(f"  {'method':12s} {'split':7s} ECE raw  ECE    cw-ECE Brier  | ECE by center 0-5")
    for m in methods:
        for k in ("image", "lesion"):
            rows = []
            for s in seeds:
                o = data[(m, k, s)]
                per_c = [ece(o["p"][o["center"] == c], o["y"][o["center"] == c]) for c in range(N_C)]
                rows.append([ece(o["p_raw"], o["y"]), ece(o["p"], o["y"]), classwise_ece(o["p"], o["y"]),
                             brier(o["p"], o["y"])] + per_c)
            r = np.mean(rows, 0)
            res["calibration"][f"{m}|{k}"] = {"ece_raw": r[0], "ece": r[1], "classwise_ece": r[2], "brier": r[3],
                                              "ece_by_center": r[4:].tolist(), "seed_sd_ece": float(np.std([x[1] for x in rows], ddof=1))}
            print(f"  {m:12s} {k:7s} {r[0]:.3f}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}  | " + " ".join(f"{x:.3f}" for x in r[4:]))

    # MC-Dropout (model A, FedAvg): raw probabilities, compared with raw softmax of the same model
    if not a.prefix:
        print("\nMC-Dropout vs softmax (model A FedAvg, raw probabilities, T = 1; mean of 5 seeds)")
        for k in ("image", "lesion"):
            r = []
            for s in seeds:
                f = os.path.join(a.runs, f"resplit_{k}_seed{s}_val", f"fedavg_r9_s{s}", "preds_mc.npz")
                if not os.path.exists(f):
                    break
                z = np.load(f)
                o = data[("fedavg_r9", k, s)]
                assert (z["target"] == o["y"]).all()
                r.append([ece(o["p_raw"], o["y"]), ece(z["probs"], o["y"]), brier(o["p_raw"], o["y"]),
                          brier(z["probs"], o["y"]), bacc(o["p_raw"], o["y"]), bacc(z["probs"], o["y"])])
            if len(r) == len(seeds):
                r = np.mean(r, 0)
                res["calibration"][f"mc_dropout|{k}"] = {"ece_softmax_raw": r[0], "ece_mc_raw": r[1], "brier_softmax_raw": r[2],
                                                         "brier_mc_raw": r[3], "ba_softmax": r[4], "ba_mc": r[5]}
                print(f"  {k:7s} ECE softmax {r[0]:.3f} vs MC {r[1]:.3f} | Brier {r[2]:.3f} vs {r[3]:.3f} | BA {r[4]:.3f} vs {r[5]:.3f}")
            else:
                print(f"  {k}: preds_mc.npz missing, run scripts/mc_dropout.py first")

    # leakage contrasts with lesion bootstrap
    print("\nLeakage and calibration (ECE, temperature-scaled; points = x100), 95% lesion-bootstrap CI")
    for m in [x for x in ("fedavg_r9", "pooled_e20") if x in methods]:
        def stat(ws):
            within, between = [], []
            for i, s in enumerate(seeds):
                oi, ol = data[(m, "image", s)], data[(m, "lesion", s)]
                wi, wl = ws[i]
                within.append(ece(oi["p"][oi["new"]], oi["y"][oi["new"]], wi[oi["new"]]) - ece(oi["p"], oi["y"], wi))
                between.append(ece(oi["p"], oi["y"], wi) - ece(ol["p"], ol["y"], wl))
            return np.mean(within), np.mean(between)
        ones = [(np.ones(len(data[(m, "image", s)]["y"])), np.ones(len(data[(m, "lesion", s)]["y"]))) for s in seeds]
        pw, pb = stat(ones)
        bs = [stat([(lesion_weights(data[(m, "image", s)], rng), lesion_weights(data[(m, "lesion", s)], rng)) for s in seeds])
              for _ in range(a.n_boot)]
        bw, bb = np.array(bs).T
        res["leakage_contrasts"][m] = {
            "within_new_minus_full": [pw, *np.percentile(bw, [2.5, 97.5]).tolist()],
            "between_image_minus_lesion": [pb, *np.percentile(bb, [2.5, 97.5]).tolist()]}
        print(f"  {m:12s} new-lesion minus full (same model): {100 * pw:+.2f} [{100 * np.percentile(bw, 2.5):+.2f}, "
              f"{100 * np.percentile(bw, 97.5):+.2f}] | image-level minus lesion-grouped model: {100 * pb:+.2f} "
              f"[{100 * np.percentile(bb, 2.5):+.2f}, {100 * np.percentile(bb, 97.5):+.2f}]")

    # ---------------- Part 2: conformal coverage under leakage (FedAvg, LAC, alpha 0.1)
    alpha = 0.1
    print("\nConformal coverage under leakage (FedAvg, LAC, alpha 0.1; mean over 5 seeds x 100 lesion splits)")
    for k in ("image", "lesion"):
        acc = {"global": {"leaked": [], "new": [], "all": []}, "clientwise": {"leaked": [], "new": [], "all": []}}
        for s in seeds:
            o = data[("fedavg_r9", k, s)]
            sc = 1 - o["p"][np.arange(len(o["y"])), o["y"]]
            for _ in range(100):
                is_cal = np.zeros(len(sc), bool)
                for c in range(N_C):
                    idx = np.flatnonzero(o["center"] == c)
                    u = np.unique(o["lesion"][idx])
                    pick = set(rng.permutation(u)[: len(u) // 2])
                    is_cal[idx[np.isin(o["lesion"][idx], list(pick))]] = True
                q_c = np.ones(N_C)
                for c in range(N_C):
                    sc_c = np.sort(sc[is_cal & (o["center"] == c)])
                    lvl = math.ceil((len(sc_c) + 1) * (1 - alpha))
                    q_c[c] = 1.0 if lvl > len(sc_c) else sc_c[lvl - 1]
                allc = np.sort(sc[is_cal])
                lvl = math.ceil((len(allc) + 1) * (1 - alpha))
                q_g = 1.0 if lvl > len(allc) else allc[lvl - 1]
                ev = ~is_cal
                for name, thr in (("global", np.full(len(sc), q_g)), ("clientwise", q_c[o["center"]])):
                    cov = sc <= thr
                    acc[name]["all"].append(cov[ev].mean())
                    if k == "image":
                        acc[name]["leaked"].append(cov[ev & ~o["new"]].mean())
                        acc[name]["new"].append(cov[ev & o["new"]].mean())
        res["conformal_leakage"][k] = {n: {g: float(np.mean(v)) for g, v in d.items() if v} for n, d in acc.items()}
        for n, d in res["conformal_leakage"][k].items():
            print(f"  {k:7s} {n:10s} " + " | ".join(f"{g}: {v:.3f}" for g, v in d.items()))

    # ---------------- Part 3: E6 unseen centers
    print("\nE6: balanced accuracy and ECE on a center when it was in the federation vs held out (mean of seeds 0-2)")
    print("  center  BA in-fed  BA unseen  drop   | ECE in-fed  ECE unseen")
    rows = []
    for h in range(N_C):
        r = []
        for s in (0, 1, 2):
            o_in = data[("fedavg_r9", "lesion", s)]
            o_out = load(a.runs, a.split_dir, "lesion", s, f"{a.prefix}fedavg_r9_loco{h}_s{s}")
            mi, mo = o_in["center"] == h, o_out["center"] == h
            r.append([bacc(o_in["p"][mi], o_in["y"][mi]), bacc(o_out["p"][mo], o_out["y"][mo]),
                      ece(o_in["p"][mi], o_in["y"][mi]), ece(o_out["p"][mo], o_out["y"][mo])])
        r = np.mean(r, 0)
        rows.append(r)
        res["e6"][str(h)] = {"ba_in_federation": r[0], "ba_unseen": r[1], "ece_in_federation": r[2], "ece_unseen": r[3]}
        print(f"  {h:6d}  {r[0]:.3f}     {r[1]:.3f}      {100 * (r[1] - r[0]):+5.1f}  | {r[2]:.3f}       {r[3]:.3f}")
    r = np.mean(rows, 0)
    print(f"  mean    {r[0]:.3f}     {r[1]:.3f}      {100 * (r[1] - r[0]):+5.1f}  | {r[2]:.3f}       {r[3]:.3f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2, default=float)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
