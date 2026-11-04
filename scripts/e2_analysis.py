"""
E2 analysis (protocol v0.10, Sections 1, 3, 9): method rankings and H3b, model A.

Rankings (RQ2). Ranking metric: balanced accuracy on the pooled test set (FLamby's benchmark metric,
as in E0 and E1); mean-of-centers balanced accuracy is reported as a secondary ranking.
  * tau_between: Kendall's tau between the image-level and the lesion-grouped ranking of the 7 methods
    (FedAvg, FedProx, SCAFFOLD, FedBN, FedAvg-FT, pooled, local-only), each ranking from the mean over
    5 seeds; 95% interval by lesion bootstrap (2,000 replicates; every split's test lesions resampled,
    the same replicate shared by all methods).
  * Noise floor (Section 9): tau between rankings from different seeds within the same split type
    (10 seed pairs per split type), compared with tau between single-seed rankings across split types
    (25 pairs), so both sides have the same single-seed noise.
H3b (RQ3). Per-center inflation(m, c) = BA of method m on center c's image-level test set minus its BA on
  center c's new-lesion subset (no center reweighting; classes with < 5 new-lesion images in that center
  dropped from both terms; classes absent from the center ignored). Tested in centers 0, 2, 3:
  mean inflation of the personalized/local methods minus that of FedAvg, centers weighted by their
  new-lesion counts, seeds averaged inside each lesion-bootstrap replicate. FedPer comes in E7; until
  then the personalized group is local-only, FedAvg-FT and FedBN, and the result is labelled interim.
Dose-response (secondary): per method, weighted least-squares slope of the per-center inflation on the
  center's leakage rate (share of its image-level test images whose lesion is in training), all six
  centers, weights = new-lesion test images; interval by the same lesion bootstrap.

E7 (--final): FedPer (fedper_r9) is added to the methods and to the personalized group, and the H3b
  result is the final, protocol test (personalized group = local-only, FedAvg-FT, FedBN, FedPer for
  model A; local-only, FedAvg-FT, FedPer-style head for model B). The E2 rankings (7 methods) stay the
  pre-specified RQ2 result; the 8-method rankings printed in this mode are supplementary.

Usage (from the repo folder):
  python scripts\\e2_analysis.py --final                      # E7, model A -> results/E7_analysis.json
  python scripts\\e2_analysis.py --final --prefix pandermfrozen_ --out results\\E7_analysis_panderm.json
  python scripts\\e2_analysis.py                              # model A
  python scripts\\e2_analysis.py --prefix pandermfrozen_ --out results\\E2_analysis_panderm.json   # model B
Writes results/E2_analysis.json and prints the tables.
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

METHODS = ["fedavg_r9", "fedprox_r9", "scaffold_r9", "fedbn_r9", "fedavgft_r9", "pooled_e20", "local_e20"]
PERSONAL = ["local_e20", "fedavgft_r9", "fedbn_r9"]          # + FedPer in E7
H3B_CENTERS = [0, 2, 3]
N_CLASSES, N_CENTERS, MIN_NEW = 8, 6, 5


def load(runs, split_dir, kind, seed, prefix):
    name = f"resplit_{kind}_seed{seed}_val"
    sp = pd.read_csv(os.path.join(split_dir, name + ".csv"))
    test = sp[sp.fold == "test"].set_index("image")
    train_les = set(sp.loc[sp.fold == "train", "lesion_id"])
    correct = {}
    base = None
    for m in METHODS:
        p = np.load(os.path.join(runs, name, f"{prefix}{m}_s{seed}", "preds.npz"))
        df = pd.DataFrame({"image": p["image"].astype(str), "center": p["center"], "target": p["target"],
                           "pred": p["logits"].argmax(1)}).set_index("image").loc[test.index]
        if base is None:
            base = df[["center", "target"]].copy()
        assert (df.target.to_numpy() == base.target.to_numpy()).all()
        correct[m] = (df.pred == df.target).to_numpy().astype(float)
    base["lesion_id"] = test.lesion_id
    base["new"] = ~base.lesion_id.isin(train_les)
    les, inv = np.unique(base.lesion_id.to_numpy(), return_inverse=True)
    return {"y": base.target.to_numpy(), "center": base.center.to_numpy(), "new": base.new.to_numpy(),
            "inv": inv, "n_les": len(les), "M": np.stack([correct[m] for m in METHODS], 1)}


def ba(d, w, mask=None, classes=None):
    """Balanced accuracy of every method (vector), images weighted by w, over the classes present."""
    mask = np.ones(len(d["y"]), bool) if mask is None else mask
    out, cls = [], (range(N_CLASSES) if classes is None else classes)
    for k in cls:
        sel = mask & (d["y"] == k)
        wk = w[sel]
        if wk.sum() > 0:
            out.append((wk[:, None] * d["M"][sel]).sum(0) / wk.sum())
    return np.mean(out, 0) if out else np.full(d["M"].shape[1], np.nan)


def ba_mean_centers(d, w):
    return np.nanmean([ba(d, w, d["center"] == c) for c in range(N_CENTERS)], 0)


def center_inflation(d, w, c):
    """Per-method inflation in center c, and the center's new-lesion count."""
    in_c = d["center"] == c
    new_c = in_c & d["new"]
    keep = [k for k in range(N_CLASSES) if (d["y"][new_c] == k).sum() >= MIN_NEW]
    if not keep:
        return np.full(d["M"].shape[1], np.nan), 0
    return ba(d, w, in_c, keep) - ba(d, w, new_c, keep), int(w[new_c].sum())


def ranks(v):
    return pd.Series(-v).rank(method="average").to_numpy()


def tau(a, b):
    return float(kendalltau(ranks(a), ranks(b)).statistic)


def summarize(data, w):
    """All statistics for one set of image weights (w[kind][seed])."""
    res = {}
    for metric, fn in (("pooled", ba), ("mean_of_centers", ba_mean_centers)):
        per = {k: {s: fn(data[k][s], w[k][s]) for s in data[k]} for k in data}
        mean = {k: np.mean(list(per[k].values()), 0) for k in per}
        seeds = sorted(data["image"])
        within = [tau(per[k][i], per[k][j]) for k in per for i, j in itertools.combinations(seeds, 2)]
        cross = [tau(per["image"][i], per["lesion"][j]) for i in seeds for j in seeds]
        res[metric] = {"mean_ba": {k: mean[k].tolist() for k in mean}, "tau_between": tau(mean["image"], mean["lesion"]),
                       "tau_cross_single_seed": float(np.mean(cross)), "tau_noise_floor": float(np.mean(within)),
                       "per_seed": {k: {s: per[k][s].tolist() for s in per[k]} for k in per}}
    # per-center inflation on the image-level splits
    infl = np.zeros((len(data["image"]), N_CENTERS, len(METHODS)))
    cnt = np.zeros((len(data["image"]), N_CENTERS))
    leak = np.zeros((len(data["image"]), N_CENTERS))
    for i, s in enumerate(sorted(data["image"])):
        d = data["image"][s]
        for c in range(N_CENTERS):
            infl[i, c], cnt[i, c] = center_inflation(d, w["image"][s], c)
            in_c = d["center"] == c
            leak[i, c] = 1 - np.sum(w["image"][s][in_c & d["new"]]) / max(np.sum(w["image"][s][in_c]), 1e-9)
    mi = np.nanmean(infl, 0)                                         # centers x methods (seed mean)
    mc = cnt.mean(0)
    lk = leak.mean(0)
    wts = np.array([mc[c] for c in H3B_CENTERS])
    per_m = np.array([np.average(mi[H3B_CENTERS, j], weights=wts) for j in range(len(METHODS))])
    pers = np.mean([per_m[METHODS.index(m)] for m in PERSONAL])
    res["h3b"] = {"inflation_by_method_centers_023": dict(zip(METHODS, per_m.tolist())),
                  "personalized_mean": float(pers), "fedavg": float(per_m[0]), "difference": float(pers - per_m[0])}
    ok = mc > 0
    slopes = {}
    for j, m in enumerate(METHODS):
        sel = ok & ~np.isnan(mi[:, j])
        X = np.c_[np.ones(sel.sum()), lk[sel]]
        W = np.diag(mc[sel])
        beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ mi[sel, j], rcond=None)[0]
        slopes[m] = float(beta[1])
    res["dose_response_slope"] = slopes
    res["per_center"] = {"inflation": {m: mi[:, j].tolist() for j, m in enumerate(METHODS)},
                         "new_lesion_images": mc.tolist(), "leakage_rate": lk.tolist()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split-dir", default=os.path.join("splits", "e1"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--prefix", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--final", action="store_true", help="E7: add FedPer, final H3b test")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join("results", "E7_analysis.json" if a.final else "E2_analysis.json")
    if a.final:
        METHODS.append("fedper_r9")
        PERSONAL.append("fedper_r9")
    if a.prefix:                                   # frozen-encoder models: no FedBN (no BatchNorm in a linear head)
        METHODS.remove("fedbn_r9")
        PERSONAL.remove("fedbn_r9")

    data = {k: {s: load(a.runs, a.split_dir, k, s, a.prefix) for s in a.seeds} for k in ("image", "lesion")}
    ones = {k: {s: np.ones(len(data[k][s]["y"])) for s in data[k]} for k in data}
    point = summarize(data, ones)

    rng = np.random.default_rng(20261103)
    keys = ["tau_between", "tau_cross_single_seed", "tau_noise_floor"]
    boot = {f"{mt}.{k}": [] for mt in ("pooled", "mean_of_centers") for k in keys}
    boot["h3b.difference"] = []
    boot.update({f"slope.{m}": [] for m in METHODS})
    for _ in range(a.n_boot):
        w = {k: {s: rng.multinomial(d["n_les"], np.full(d["n_les"], 1 / d["n_les"]))[d["inv"]].astype(float)
                 for s, d in data[k].items()} for k in data}
        r = summarize(data, w)
        for mt in ("pooled", "mean_of_centers"):
            for k in keys:
                boot[f"{mt}.{k}"].append(r[mt][k])
        boot["h3b.difference"].append(r["h3b"]["difference"])
        for m in METHODS:
            boot[f"slope.{m}"].append(r["dose_response_slope"][m])
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boot.items()}

    names = {"fedavg_r9": "FedAvg", "fedprox_r9": "FedProx", "scaffold_r9": "SCAFFOLD", "fedbn_r9": "FedBN",
             "fedavgft_r9": "FedAvg-FT", "pooled_e20": "Pooled", "local_e20": "Local-only",
             "fedper_r9": "FedPer"}
    for mt in ("pooled", "mean_of_centers"):
        p = point[mt]
        print(f"\n=== Ranking by {mt} balanced accuracy (mean of {len(a.seeds)} seeds) ===")
        ri, rl = ranks(np.array(p["mean_ba"]["image"])), ranks(np.array(p["mean_ba"]["lesion"]))
        for j, m in enumerate(METHODS):
            print(f"  {names[m]:11s} image {p['mean_ba']['image'][j]:.4f} (rank {ri[j]:.0f}) | "
                  f"lesion {p['mean_ba']['lesion'][j]:.4f} (rank {rl[j]:.0f})")
        print(f"  tau between split types (seed-mean rankings): {p['tau_between']:+.3f} "
              f"[{ci[mt + '.tau_between'][0]:+.3f}, {ci[mt + '.tau_between'][1]:+.3f}]")
        print(f"  single-seed tau across split types: {p['tau_cross_single_seed']:+.3f} "
              f"[{ci[mt + '.tau_cross_single_seed'][0]:+.3f}, {ci[mt + '.tau_cross_single_seed'][1]:+.3f}]"
              f"  vs noise floor (within split type): {p['tau_noise_floor']:+.3f} "
              f"[{ci[mt + '.tau_noise_floor'][0]:+.3f}, {ci[mt + '.tau_noise_floor'][1]:+.3f}]")

    h = point["h3b"]
    label = "final" if a.final else "interim"
    print(f"\n=== H3b ({'final, with FedPer' if a.final else 'interim, without FedPer'}): inflation in centers 0, 2, 3, weighted by new-lesion counts ===")
    for m, v in h["inflation_by_method_centers_023"].items():
        print(f"  {names[m]:11s} {100 * v:+.2f} pts")
    print(f"  personalized/local mean {100 * h['personalized_mean']:+.2f} - FedAvg {100 * h['fedavg']:+.2f} = "
          f"{100 * h['difference']:+.2f} pts, 95% CI [{100 * ci['h3b.difference'][0]:+.2f}, {100 * ci['h3b.difference'][1]:+.2f}]"
          f"  -> {'supported' if ci['h3b.difference'][0] > 0 else 'not supported'} ({label})")
    pc = point["per_center"]
    print("\n=== Per-center inflation (pts) and leakage rate ===")
    print("  center      " + " ".join(f"{c:>8d}" for c in range(N_CENTERS)))
    print("  leakage     " + " ".join(f"{100 * x:7.1f}%" for x in pc["leakage_rate"]))
    print("  new images  " + " ".join(f"{x:8.0f}" for x in pc["new_lesion_images"]))
    for m in METHODS:
        print(f"  {names[m]:11s} " + " ".join(f"{100 * x:+8.1f}" if np.isfinite(x) else "     n/a" for x in pc["inflation"][m]))
    print("\n=== Dose-response: WLS slope of per-center inflation on leakage rate (pts per 100% leakage) ===")
    for m in METHODS:
        print(f"  {names[m]:11s} {100 * point['dose_response_slope'][m]:+.1f} "
              f"[{100 * ci['slope.' + m][0]:+.1f}, {100 * ci['slope.' + m][1]:+.1f}]")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"point": point, "ci95": ci, "methods": METHODS, "personalized": PERSONAL, "final": a.final, "seeds": a.seeds, "n_boot": a.n_boot},
              open(a.out, "w"), indent=2, default=float)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
