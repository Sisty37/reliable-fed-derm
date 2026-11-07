"""
Post hoc sensitivity analyses of the conformal results (protocol v0.12, Section 12, 7 Nov 2026).
No new training: uses the predictions of E5 (lesion-grouped FedAvg, seeds 0-4) and E6 (leave-one-center-out
FedAvg, 6 held-out centers x seeds 0-2), models A and B, LAC score, alpha = 0.1, same temperatures as E5/E6.

  1. Monte Carlo stability: worst-center lower-tail coverage (q10) from 5 independent sets of 100 splits and
     from 1,000 splits.
  2. Bootstrap keeping lesion identity: in each replicate the resampled copies of one test lesion are kept
     together (same calibration/evaluation half). 2,000 replicates x 20 splits (as pre-specified) and
     500 replicates x 100 splits (Monte Carlo check of the interval).
  3. One image per lesion: after the lesion-wise halves are drawn, one random image per lesion is kept in both
     halves, so calibration and evaluation units are independent lesions (target: a random image of a new
     lesion). All methods; EB-ShrinkFCP gain with a lesion bootstrap (500 x 20).
  4. Exact quantiles: global and FCP thresholds from the exact order statistic instead of the 1,000-bin
     histogram.
  5. Calibration-only variability: for each model, 5 fixed evaluation halves; for each, 100 calibration sets
     drawn by lesion bootstrap from the other half; coverage on the fixed evaluation half. The q10 over
     calibration draws then reflects threshold variability, not evaluation-sample noise.
  6. Label-shift-weighted comparator (LSW): per center, weighted quantile of the pooled calibration scores with
     weights pi_c(y)/pi_pool(y) from calibration class counts (weighted split conformal under label shift,
     Podkopaev & Ramdas 2021; federated: clients share per-class score histograms). The test-point weight is
     bounded by max_y pi_c(y)/pi_pool(y) (conservative).

Usage (from the repo folder):  python scripts/conformal_sensitivity.py
Output: results/Conformal_sensitivity.json (+ printed tables).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_conformal as E  # noqa: E402

ALPHA = 0.1
METHODS = ["global", "clientwise", "fcp", "eb", "lsw"]


def exact_quantile(s, rank):
    if rank > len(s):
        return 1.0
    return float(np.sort(s)[max(rank, 1) - 1])


def lsw_thresholds(cal_s, cal_y, alpha):
    """Label-shift-weighted federated threshold per center (per-class 1,000-bin histograms)."""
    K = E.N_CENTERS
    hist = np.zeros((K, E.N_CLASSES, len(E.BINS) - 1))
    for c in range(K):
        for k in range(E.N_CLASSES):
            hist[c, k] = np.histogram(np.clip(cal_s[c][cal_y[c] == k], 0, 1), E.BINS)[0]
    cls_c = hist.sum(2)                                    # centers x classes
    pi_pool = cls_c.sum(0) / cls_c.sum()
    pooled = hist.sum(0)                                   # classes x bins
    thr = np.ones(K)
    for c in range(K):
        if cls_c[c].sum() == 0:
            continue
        pi_c = cls_c[c] / cls_c[c].sum()
        w = np.where(pi_pool > 0, pi_c / np.maximum(pi_pool, 1e-12), 0.0)
        mass = (w[:, None] * pooled).sum(0)                # weighted counts per bin
        total = mass.sum() + w.max()                       # + conservative test-point weight
        cum = np.cumsum(mass) / total
        hit = np.flatnonzero(cum >= 1 - alpha)
        thr[c] = float(E.BINS[1:][hit[0]]) if len(hit) else 1.0
    return thr


def thresholds(cal_s, cal_y, n_c, alpha, methods, exact=False):
    m_c = [len(s) for s in cal_s]
    m = sum(m_c)
    allc = np.concatenate(cal_s)
    counts = np.histogram(np.clip(allc, 0, 1), E.BINS)[0]
    q_c = np.array([E.level_quantile(s, math.ceil((len(s) + 1) * (1 - alpha)) / max(len(s), 1)) for s in cal_s])
    gq = (lambda r: exact_quantile(allc, r)) if exact else (lambda r: E.hist_quantile(counts, r))
    t = {}
    if "global" in methods:
        t["global"] = np.full(E.N_CENTERS, gq(math.ceil((m + 1) * (1 - alpha))))
    if "clientwise" in methods:
        t["clientwise"] = q_c
    if "fcp" in methods:
        t["fcp"] = np.full(E.N_CENTERS, gq(math.ceil((m + E.N_CENTERS) * (1 - alpha))))
    if "eb" in methods:
        qv = [E.eb_inputs(cs, alpha, len(cs), int(nc)) for cs, nc in zip(cal_s, n_c)]
        t["eb"] = E.eb_thresholds(np.array([x[0] for x in qv]), np.array([x[1] for x in qv]))
    if "lsw" in methods:
        t["lsw"] = lsw_thresholds(cal_s, cal_y, alpha)
    return t


class U:
    """Test set of one model, organised by center and lesion."""

    def __init__(self, o, lesion=None):
        self.s, self.all = o["scores"]["lac"]
        self.y, self.center = o["y"], o["center"]
        les = o["lesion"] if lesion is None else lesion
        self.groups = []                                   # per center: list of image-index arrays, one per lesion
        for c in range(E.N_CENTERS):
            idx = np.flatnonzero(self.center == c)
            _, inv = np.unique(les[idx], return_inverse=True)
            order = np.argsort(inv, kind="stable")
            bounds = np.cumsum(np.bincount(inv))[:-1]
            self.groups.append(np.split(idx[order], bounds))


def one_split(u, rng, methods, one_per_lesion=False, exact=False, fixed_eval=None, boot_cal=None):
    cal_s, cal_y, n_c, ev = [], [], [], []
    for c, g in enumerate(u.groups):
        nl = len(g)
        if fixed_eval is not None:
            ev_l, cal_l = fixed_eval[c]
            cal_l = cal_l[rng.integers(0, len(cal_l), len(cal_l))] if boot_cal else cal_l
        else:
            perm = rng.permutation(nl)
            cal_l, ev_l = perm[: nl // 2], perm[nl // 2:]
        pick = (lambda L: np.array([g[i][rng.integers(len(g[i]))] for i in L], int)) if one_per_lesion \
            else (lambda L: np.concatenate([g[i] for i in L]) if len(L) else np.array([], int))
        ci, ei = pick(cal_l), pick(ev_l)
        cal_s.append(u.s[ci])
        cal_y.append(u.y[ci])
        n_c.append(len(cal_l))
        ev.append(ei)
    t = thresholds(cal_s, cal_y, n_c, ALPHA, methods, exact)
    out = {}
    for name, th in t.items():
        cov = np.array([np.mean(u.s[e] <= th[c]) if len(e) else np.nan for c, e in enumerate(ev)])
        size = np.array([np.mean((u.all[e] <= th[c]).sum(1)) if len(e) else np.nan for c, e in enumerate(ev)])
        out[name] = (cov, size)
    return out


def summarize_runs(u, rng, n, methods, **kw):
    acc = {}
    for _ in range(n):
        for k, (c, s) in one_split(u, rng, methods, **kw).items():
            acc.setdefault(k, ([], []))
            acc[k][0].append(c)
            acc[k][1].append(s)
    return {k: E.summarize(np.array(c), np.array(s)) for k, (c, s) in acc.items()}


def mean_over_units(per):
    return {k: {"worst_center_q10": float(np.mean([p[k]["worst_center_q10"] for p in per])),
                "mean_coverage": np.mean([p[k]["mean_coverage"] for p in per], 0).tolist(),
                "lower_tail_q10": np.mean([p[k]["lower_tail_q10"] for p in per], 0).tolist(),
                "mean_set_size": np.mean([p[k]["mean_set_size"] for p in per], 0).tolist(),
                "set_size_equal_weight": float(np.mean([p[k]["set_size_equal_weight"] for p in per]))}
            for k in per[0]}


def boot_gain(data, units, rng, n_boot, n_splits, one_per_lesion=False):
    """Paired lesion bootstrap of the EB gain over client-wise and global; copies of a lesion stay together."""
    by_seed = {}
    for s, r in units:
        by_seed.setdefault(s, []).append((s, r))
    members = {}
    for s in by_seed:                       # all models of one seed share the same test images in the same order
        o = data[by_seed[s][0]]["test"]
        _, inv = np.unique(o["lesion"], return_inverse=True)
        order = np.argsort(inv, kind="stable")
        members[s] = np.split(order, np.cumsum(np.bincount(inv))[:-1])
        for u in by_seed[s]:
            assert (data[u]["test"]["lesion"] == o["lesion"]).all()
    gains = {"clientwise": [], "global": []}
    for b in range(n_boot):
        g = {"clientwise": [], "global": []}
        for s, us in by_seed.items():
            mem = members[s]
            nl = len(mem)
            chosen = rng.integers(0, nl, nl)
            take = np.concatenate([mem[l] for l in chosen])
            orig = np.concatenate([np.full(len(mem[l]), l) for l in chosen])     # ORIGINAL lesion identity
            for u in us:
                o = data[u]["test"]
                ob = {"scores": {"lac": (o["scores"]["lac"][0][take], o["scores"]["lac"][1][take])},
                      "y": o["y"][take], "center": o["center"][take], "lesion": orig}
                r = summarize_runs(U(ob), rng, n_splits, ["global", "clientwise", "eb"], one_per_lesion=one_per_lesion)
                for ref in g:
                    g[ref].append(r["eb"]["worst_center_q10"] - r[ref]["worst_center_q10"])
        for ref in g:
            gains[ref].append(np.mean(g[ref]))
    return {ref: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for ref, v in gains.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split-dir", default=os.path.join("splits", "e1"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="small numbers of replicates (pipeline test)")
    ap.add_argument("--out", default=os.path.join("results", "Conformal_sensitivity.json"))
    a = ap.parse_args()
    nb = 40 if a.quick else a.n_boot
    res = {}
    for model, prefix in (("A", ""), ("B", "pandermfrozen_")):
        for exp in ("E6_loco", "E5"):
            t0 = time.time()
            rng = np.random.default_rng(20261107)
            if exp == "E6_loco":
                units = [(s, f"{prefix}fedavg_r9_loco{h}_s{s}") for s in (0, 1, 2) for h in range(E.N_CENTERS)]
            else:
                units = [(s, f"{prefix}fedavg_r9_s{s}") for s in range(5)]
            data = {u: E.load(a.runs, a.split_dir, u[0], u[1]) for u in units}
            for u in units:
                data[u]["test"]["y"] = data[u]["test"]["y"]
            us = {u: U(data[u]["test"]) for u in units}
            key = f"{model}/{exp}"
            r = {}
            # 1. Monte Carlo stability
            r["mc_100x5"] = [mean_over_units([summarize_runs(us[u], rng, 100, METHODS) for u in units]) for _ in range(5 if not a.quick else 2)]
            r["mc_1000"] = mean_over_units([summarize_runs(us[u], rng, 1000 if not a.quick else 100, METHODS) for u in units])
            # 3. one image per lesion
            r["one_per_lesion"] = mean_over_units([summarize_runs(us[u], rng, 1000 if not a.quick else 100, METHODS,
                                                                  one_per_lesion=True) for u in units])
            # 4. exact quantiles
            r["exact_quantiles"] = mean_over_units([summarize_runs(us[u], rng, 1000 if not a.quick else 100,
                                                                   ["global", "fcp"], exact=True) for u in units])
            # 5. calibration-only variability on fixed evaluation halves
            per = []
            for u in units:
                for _ in range(5 if not a.quick else 2):
                    fe = []
                    for g in us[u].groups:
                        perm = rng.permutation(len(g))
                        fe.append((perm[len(g) // 2:], perm[: len(g) // 2]))
                    per.append(summarize_runs(us[u], rng, 100, ["global", "clientwise", "eb"], fixed_eval=fe, boot_cal=True))
            r["calibration_only"] = mean_over_units(per)
            # 2. bootstrap keeping lesion identity (+ one-image-per-lesion version)
            if exp == "E6_loco":
                r["boot_identity_2000x20"] = boot_gain(data, units, rng, nb, 20)
                r["boot_identity_500x100"] = boot_gain(data, units, rng, max(nb // 4, 10), 100)
                r["boot_one_per_lesion_500x20"] = boot_gain(data, units, rng, max(nb // 4, 10), 20, one_per_lesion=True)
            res[key] = r
            m = r["mc_1000"]
            print(f"\n=== model {model}, {exp} ({len(units)} models, {time.time() - t0:.0f} s) ===")
            print(f"  {'method':12s} {'q10 (1000 splits)':>18s} {'5 x 100 splits':>26s} {'1 img/lesion':>13s} "
                  f"{'exact q':>8s} {'cal-only q10':>13s} {'min mean cov':>13s} {'size':>6s}")
            for k in METHODS:
                mc = [x[k]["worst_center_q10"] for x in r["mc_100x5"]]
                ex = r["exact_quantiles"].get(k, {}).get("worst_center_q10", np.nan)
                co = r["calibration_only"].get(k, {}).get("worst_center_q10", np.nan)
                print(f"  {k:12s} {m[k]['worst_center_q10']:18.3f} {min(mc):12.3f} - {max(mc):.3f}   "
                      f"{r['one_per_lesion'][k]['worst_center_q10']:13.3f} {ex:8.3f} {co:13.3f} "
                      f"{min(m[k]['mean_coverage']):13.3f} {m[k]['set_size_equal_weight']:6.2f}")
            if exp == "E6_loco":
                for name in ("boot_identity_2000x20", "boot_identity_500x100", "boot_one_per_lesion_500x20"):
                    print(f"  EB gain CI, {name}: vs client-wise [{100 * r[name]['clientwise'][0]:+.2f}, "
                          f"{100 * r[name]['clientwise'][1]:+.2f}], vs global [{100 * r[name]['global'][0]:+.2f}, "
                          f"{100 * r[name]['global'][1]:+.2f}] pts")
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            json.dump(res, open(a.out, "w"), indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
