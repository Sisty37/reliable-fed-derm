"""
E5 / E5b (protocol v0.10, Section 5): federated conformal prediction on the real centers and the
proposed shrinkage federated conformal calibration (ShrinkFCP, RQ4).

Models: lesion-grouped FedAvg models (9 rounds), seeds 0-4; model A (--prefix "") and model B
(--prefix pandermfrozen_). Evaluated separately; the RQ4 success criterion must hold for each.

Probabilities: softmax with a global temperature. Each client fits its own temperature on its
validation lesions (NLL); the global temperature is their mean weighted by validation images.

Scores: LAC (1 - p_y; primary) and APS (non-randomized: probability mass of all classes ranked at or
above the true class). Prediction set of an image = all classes whose score is <= the threshold (empty
sets allowed). Target coverage 1 - alpha with alpha = 0.1 (primary), 0.05 and 0.2.

Calibration / evaluation: within each center's test set, lesions are split in halves at random
(calibration / evaluation), 100 times. Methods (threshold per center c):
  global          pooled calibration scores, level ceil((m+1)(1-a))/m, from 1,000-bin histograms
                  that each client shares (no raw scores leave a client)
  client-wise     the center's own scores, level ceil((m_c+1)(1-a))/m_c (threshold 1 if above 1)
  sample-weighted sum_c (m_c/m) q_c, the same threshold for every center
  FCP             as global, level ceil((m+K)(1-a))/m (Lu et al., 2023, authors' code)
  ShrinkFCP       w_c q_c + (1 - w_c) q_global, w_c = n_c / (n_c + k), n_c = calibration lesions of c,
                  k = median over centers of n_c (primary rule)
Personalized FCP (Min et al., 2025) is not included yet (see results/E5_implementation_notes.md).

RQ4 metric: per center, the 10th percentile of coverage over the 100 splits (lower-tail coverage) and
P(coverage < 0.85); primary = worst-center lower-tail coverage (min over centers), averaged over seeds.
Reference = global or client-wise, whichever has the higher primary metric. Success (alpha 0.1, LAC):
ShrinkFCP raises the primary metric by >= 2 points with the lesion-bootstrap interval of the gain above
zero (2,000 replicates; each replicate resamples every seed's test lesions and redraws 20 calibration /
evaluation splits shared by all methods; seeds averaged inside the replicate), mean set size (equal
weight per center) <= 1.10 x the reference's, and no center's mean coverage below 0.89.
Sensitivity: the primary metric over k in {0, 10, 25, 50, 100, 200, 500, inf} on the test sets, and a
data-driven k chosen on validation lesions (each center's validation lesions split 50/50, 100 times,
objective = the primary metric), then applied to the test sets.

Usage (from the repo folder):
  python scripts\\e5_conformal.py                                   # model A
  python scripts\\e5_conformal.py --prefix pandermfrozen_            # model B
Writes results/E5_conformal[_<prefix>].json and prints the tables.
"""
import argparse
import json
import math
import os

import numpy as np
import pandas as pd

N_CENTERS, N_CLASSES = 6, 8
K_GRID = [0, 10, 25, 50, 100, 200, 500, math.inf]
METHODS = ["global", "clientwise", "weighted", "fcp", "shrink"]
BINS = np.linspace(0.0, 1.0, 1001)


# ----------------------------------------------------------------------------- data
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 600))
    nll = [-np.log(softmax(logits / t)[np.arange(len(y)), y] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])


def scores(p, y):
    """LAC and APS score of the true class, and per-class scores (for set sizes)."""
    lac_all = 1.0 - p
    order = np.argsort(-p, 1)
    cum_sorted = np.cumsum(np.take_along_axis(p, order, 1), 1)
    aps_all = np.empty_like(p)
    np.put_along_axis(aps_all, order, cum_sorted, 1)
    idx = np.arange(len(y))
    return {"lac": (lac_all[idx, y], lac_all), "aps": (aps_all[idx, y], aps_all)}


def load(runs, split_dir, seed, prefix):
    name = f"resplit_lesion_seed{seed}_val"
    sp = pd.read_csv(os.path.join(split_dir, name + ".csv")).set_index("image")
    d = os.path.join(runs, name, f"{prefix}fedavg_r9_s{seed}")
    out = {}
    for fold, f in (("test", "preds.npz"), ("val", "preds_val.npz")):
        p = np.load(os.path.join(d, f))
        out[fold] = {"logits": p["logits"].astype(np.float64), "y": p["target"].astype(int),
                     "center": p["center"].astype(int),
                     "lesion": sp.loc[p["image"].astype(str), "lesion_id"].to_numpy()}
    v = out["val"]
    temps, sizes = [], []
    for c in range(N_CENTERS):
        m = v["center"] == c
        temps.append(fit_temperature(v["logits"][m], v["y"][m]))
        sizes.append(int(m.sum()))
    t_glob = float(np.average(temps, weights=sizes))
    for fold in out:
        o = out[fold]
        o["p"] = softmax(o["logits"] / t_glob)
        o["scores"] = scores(o["p"], o["y"])
    out["temperature"] = {"per_center": temps, "global": t_glob}
    return out


# ----------------------------------------------------------------------------- conformal core
def level_quantile(s, level):
    if level > 1:
        return 1.0                                    # threshold covering every class
    return float(np.sort(s)[max(int(math.ceil(level * len(s))) - 1, 0)])


def hist_quantile(counts, rank):
    """Upper bin edge where the cumulative count first reaches `rank` (conservative)."""
    total = counts.sum()
    if rank > total:
        return 1.0
    return float(BINS[1:][np.searchsorted(np.cumsum(counts), rank)])


class Unit:
    """One test (or validation) set prepared for repeated lesion-wise splitting."""

    def __init__(self, o, score, unit_lesion=None):
        self.s, self.all = o["scores"][score]
        self.center = o["center"]
        les = o["lesion"] if unit_lesion is None else unit_lesion
        self.by_center = []
        for c in range(N_CENTERS):
            idx = np.flatnonzero(self.center == c)
            u, inv = np.unique(les[idx], return_inverse=True)
            self.by_center.append((idx, inv, len(u)))


def split_eval(unit, rng, alpha, ks, methods=METHODS, k_rule=True):
    """One random lesion-wise calibration/evaluation split; returns coverage and set size per center."""
    cal_s, ev_idx, m_c, n_c, counts = [], [], [], [], []
    for idx, inv, nl in unit.by_center:
        pick = rng.permutation(nl)[: nl // 2]
        is_cal = np.isin(inv, pick)
        s = unit.s[idx[is_cal]]
        cal_s.append(s)
        ev_idx.append(idx[~is_cal])
        m_c.append(len(s))
        n_c.append(nl // 2)
        counts.append(np.histogram(np.clip(s, 0, 1), BINS)[0])
    m = sum(m_c)
    counts = np.sum(counts, 0)
    q_c = np.array([level_quantile(s, math.ceil((len(s) + 1) * (1 - alpha)) / max(len(s), 1)) for s in cal_s])
    q_g = hist_quantile(counts, math.ceil((m + 1) * (1 - alpha)))
    thr = {}
    if "global" in methods:
        thr["global"] = np.full(N_CENTERS, q_g)
    if "clientwise" in methods:
        thr["clientwise"] = q_c
    if "weighted" in methods:
        thr["weighted"] = np.full(N_CENTERS, np.dot(m_c, q_c) / m)
    if "fcp" in methods:
        thr["fcp"] = np.full(N_CENTERS, hist_quantile(counts, math.ceil((m + N_CENTERS) * (1 - alpha))))
    n_c = np.array(n_c, float)
    if k_rule and "shrink" in methods:
        k = float(np.median(n_c))
        w = n_c / (n_c + k)
        thr["shrink"] = w * q_c + (1 - w) * q_g
    for k in ks:
        w = np.zeros(N_CENTERS) if k == math.inf else n_c / (n_c + k) if k > 0 else np.ones(N_CENTERS)
        thr[f"shrink_k{k}"] = w * q_c + (1 - w) * q_g
    res = {}
    for name, t in thr.items():
        cov = np.array([np.mean(unit.s[e] <= t[c]) if len(e) else np.nan for c, e in enumerate(ev_idx)])
        size = np.array([np.mean((unit.all[e] <= t[c]).sum(1)) if len(e) else np.nan for c, e in enumerate(ev_idx)])
        res[name] = (cov, size)
    return res


def summarize(cov, size):
    """cov, size: splits x centers. Per-center and primary statistics."""
    q10 = np.nanpercentile(cov, 10, axis=0)
    return {"lower_tail_q10": q10.tolist(), "p_below_085": np.nanmean(cov < 0.85, 0).tolist(),
            "mean_coverage": np.nanmean(cov, 0).tolist(), "mean_set_size": np.nanmean(size, 0).tolist(),
            "worst_center_q10": float(np.nanmin(q10)), "set_size_equal_weight": float(np.nanmean(np.nanmean(size, 0)))}


def run_splits(unit, rng, alpha, n_splits, ks=(), methods=METHODS, k_rule=True):
    acc = {}
    for _ in range(n_splits):
        for name, (c, s) in split_eval(unit, rng, alpha, ks, methods, k_rule).items():
            acc.setdefault(name, ([], []))
            acc[name][0].append(c)
            acc[name][1].append(s)
    return {n: summarize(np.array(c), np.array(s)) for n, (c, s) in acc.items()}


def seed_mean(per_seed):
    out = {}
    for name in per_seed[0]:
        out[name] = {k: (np.mean([ps[name][k] for ps in per_seed], 0).tolist()
                         if isinstance(per_seed[0][name][k], list) else float(np.mean([ps[name][k] for ps in per_seed])))
                     for k in per_seed[0][name]}
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split-dir", default=os.path.join("splits", "e1"))
    ap.add_argument("--prefix", default="")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-splits", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or os.path.join("results", f"E5_conformal{'_' + a.prefix.rstrip('_') if a.prefix else ''}.json")
    rng = np.random.default_rng(20261103)
    data = {s: load(a.runs, a.split_dir, s, a.prefix) for s in a.seeds}
    res = {"temperature": {s: data[s]["temperature"] for s in a.seeds}}

    # point estimates, all scores and alphas
    for score in ("lac", "aps"):
        for alpha in (0.05, 0.1, 0.2):
            per_seed = [run_splits(Unit(data[s]["test"], score), rng, alpha, a.n_splits, ks=K_GRID) for s in a.seeds]
            res[f"{score}_a{alpha}"] = seed_mean(per_seed)

    # primary: LAC, alpha 0.1
    prim = res["lac_a0.1"]
    ref = "global" if prim["global"]["worst_center_q10"] >= prim["clientwise"]["worst_center_q10"] else "clientwise"
    gain = prim["shrink"]["worst_center_q10"] - prim[ref]["worst_center_q10"]

    # data-driven k on validation lesions (sensitivity)
    k_choice = {}
    for s in a.seeds:
        val = run_splits(Unit(data[s]["val"], "lac"), rng, 0.1, a.n_splits, ks=K_GRID, methods=[], k_rule=False)
        best = max(K_GRID, key=lambda k: (val[f"shrink_k{k}"]["worst_center_q10"], -K_GRID.index(k)))
        k_choice[s] = best
    dd = []
    for s in a.seeds:
        r = run_splits(Unit(data[s]["test"], "lac"), rng, 0.1, a.n_splits, ks=[k_choice[s]], methods=[], k_rule=False)
        dd.append(r[f"shrink_k{k_choice[s]}"]["worst_center_q10"])
    res["data_driven_k"] = {"k_by_seed": {s: str(k) for s, k in k_choice.items()}, "worst_center_q10_test": float(np.mean(dd))}

    # lesion bootstrap of the gain (paired: same resample and splits for both methods)
    prep = {}
    for s in a.seeds:                                   # lesion -> images, for fast resampling
        o = data[s]["test"]
        _, inv = np.unique(o["lesion"], return_inverse=True)
        order = np.argsort(inv, kind="stable")
        lengths = np.bincount(inv)
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        prep[s] = (o, order, lengths, starts)
    boots = []
    for b in range(a.n_boot):
        g = []
        for s in a.seeds:
            o, order, lengths, starts = prep[s]
            chosen = rng.integers(0, len(lengths), len(lengths))      # lesions drawn with replacement
            ln = lengths[chosen]
            offs = np.arange(ln.sum()) - np.repeat(np.cumsum(ln) - ln, ln)
            take = order[np.repeat(starts[chosen], ln) + offs]
            ob = {"scores": {"lac": (o["scores"]["lac"][0][take], o["scores"]["lac"][1][take])},
                  "center": o["center"][take], "lesion": np.repeat(np.arange(len(chosen)), ln)}
            r = run_splits(Unit(ob, "lac"), rng, 0.1, a.boot_splits, methods=["global", "clientwise", "shrink"])
            g.append(r["shrink"]["worst_center_q10"] - r[ref]["worst_center_q10"])
        boots.append(np.mean(g))
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{a.n_boot}")
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    size_ok = prim["shrink"]["set_size_equal_weight"] <= 1.10 * prim[ref]["set_size_equal_weight"]
    cov_ok = min(prim["shrink"]["mean_coverage"]) >= 0.89
    success = gain >= 0.02 and ci[0] > 0 and size_ok and cov_ok
    res["rq4"] = {"reference": ref, "gain": gain, "ci95": ci, "set_size_ok": size_ok, "coverage_ok": cov_ok,
                  "success": success}

    # print
    names = {"global": "Global", "clientwise": "Client-wise", "weighted": "Sample-weighted", "fcp": "FCP",
             "shrink": "ShrinkFCP (k = median n_c)"}
    print(f"\nModel {'B (' + a.prefix + ')' if a.prefix else 'A'}; global temperature by seed: "
          + ", ".join(f"{res['temperature'][s]['global']:.2f}" for s in a.seeds))
    for key in ("lac_a0.1", "aps_a0.1", "lac_a0.05", "lac_a0.2"):
        print(f"\n=== {key.replace('_a', ', alpha ')} (mean of {len(a.seeds)} seeds, {a.n_splits} splits each) ===")
        print(f"  {'method':28s} worst q10 | lower-tail q10 by center (0-5)            | min mean cov | set size")
        for mth in METHODS:
            r = res[key][mth]
            print(f"  {names[mth]:28s} {r['worst_center_q10']:.3f}    | "
                  + " ".join(f"{x:.3f}" for x in r["lower_tail_q10"])
                  + f" | {min(r['mean_coverage']):.3f}        | {r['set_size_equal_weight']:.2f}")
    print("\n=== k curve (LAC, alpha 0.1): worst-center lower-tail coverage ===")
    print("  " + "  ".join(f"k={k}: {res['lac_a0.1'][f'shrink_k{k}']['worst_center_q10']:.3f}" for k in K_GRID))
    print(f"  data-driven k by seed {res['data_driven_k']['k_by_seed']} -> test {res['data_driven_k']['worst_center_q10_test']:.3f}")
    r = res["rq4"]
    print(f"\n=== RQ4 (LAC, alpha 0.1): reference = {r['reference']} ===")
    print(f"  gain {100 * r['gain']:+.2f} pts, 95% CI [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] | set size ok: {size_ok}"
          f" | all centers' mean coverage >= 0.89: {cov_ok}  ->  {'SUCCESS' if success else 'NOT MET'}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2, default=lambda x: str(x))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
