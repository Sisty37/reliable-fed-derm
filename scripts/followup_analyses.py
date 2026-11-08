"""
Analyses requested by the follow-up review of 8 November 2026 (post hoc; no new training).

1. LOCO conformal, held-out-only view: client c evaluated only with the three models trained without c
   (seeds 0-2), 1,000 lesion-wise draws per model; mean coverage, q10 and set size for global, FCP,
   client-wise and EB-ShrinkFCP. Also the two aggregations of the all-client view: the mean over models
   of each model's worst-client q10 (used in the paper) and the worst client of the model-averaged q10.
2. Eq. (1) support: number of client-class cells (and images) of the full test term without new-lesion
   images, and the apparent inflation when the full term is restricted to the same cells (common support).
3. Calibration-budget caps: realized calibration lesions per client at caps 10, 25, 50, 100, and which
   client determines the worst-client q10 (share of models), 300 draws per model.
4. E8: seed-specific exposure effects, target counts and sibling-count distribution.

Usage (from the repo folder):  python scripts/followup_analyses.py
Output: results/Followup_analyses.json
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_conformal as E  # noqa: E402
import e8_analysis as E8  # noqa: E402
import gate2 as G  # noqa: E402

ALPHA = 0.1
RULES = ("global", "fcp", "clientwise", "eb")


def loco_heldout(n_draws):
    out = {}
    for model, prefix in (("A", ""), ("B", "pandermfrozen_")):
        per = {r: {"mean_cov": [], "q10": [], "size": []} for r in RULES}
        worst_each, q10_by_model = {r: [] for r in RULES}, {r: [] for r in RULES}
        for h in range(6):
            acc = {r: {"mean_cov": [], "q10": [], "size": []} for r in RULES}
            for s in (0, 1, 2):
                u = E.Unit(E.load("runs", os.path.join("splits", "e1"), s, f"{prefix}fedavg_r9_loco{h}_s{s}")["test"], "lac")
                rng = np.random.default_rng(20261109 + 10 * h + s)
                res = E.run_splits(u, rng, ALPHA, n_draws, methods=RULES, k_rule=False)
                for r in RULES:
                    acc[r]["mean_cov"].append(res[r]["mean_coverage"][h])
                    acc[r]["q10"].append(res[r]["lower_tail_q10"][h])
                    acc[r]["size"].append(res[r]["mean_set_size"][h])
                    worst_each[r].append(res[r]["worst_center_q10"])
                    q10_by_model[r].append(res[r]["lower_tail_q10"])
            for r in RULES:
                for k in per[r]:
                    per[r][k].append(float(np.mean(acc[r][k])))
        out[model] = {"heldout_only_by_client": per,
                      "mean_of_model_worst_q10": {r: float(np.mean(worst_each[r])) for r in RULES},
                      "worst_of_model_averaged_q10": {r: float(np.min(np.mean(q10_by_model[r], 0))) for r in RULES},
                      "n_draws": n_draws}
        print(f"model {model} held-out only (client: client-wise / EB / global mean cov): " + "  ".join(
            f"{c}: {per['clientwise']['mean_cov'][c]:.3f}/{per['eb']['mean_cov'][c]:.3f}/{per['global']['mean_cov'][c]:.3f}"
            for c in range(6)))
    return out


def eq1_support():
    out = {}
    for label, tag in (("A FedAvg", "fedavg_r9"), ("A-frozen FedAvg", "effb0frozen_fedavg_r9"),
                       ("B FedAvg", "pandermfrozen_fedavg_r9"), ("A pooled", "pooled_e20")):
        rows = []
        for s in (0, 1, 2):
            name = f"resplit_image_seed{s}_val"
            df = G.load_run(os.path.join("runs", name, f"{tag}_s{s}"), os.path.join("splits", "e1", name + ".csv"))
            cls = G.kept_classes(df)
            w = np.ones(len(df))
            f, n, _ = G.terms(df, w, cls)
            # common support: full term restricted to client-class cells with new-lesion images
            full_cs, cells_out, imgs_out, cells_all = [], 0, 0, 0
            for k in cls:
                sub = df[df.target == k]
                has_new = sub.groupby("center").new.any()
                keep = sub.center.map(has_new).to_numpy()
                cells_all += len(has_new)
                cells_out += int((~has_new).sum())
                imgs_out += int((~keep).sum())
                full_cs.append(sub.correct.to_numpy()[keep].mean())
            rows.append({"seed": s, "inflation": f - n, "inflation_common_support": float(np.mean(full_cs)) - n,
                         "cells_without_new": cells_out, "cells_total": cells_all, "images_outside_support": imgs_out,
                         "test_images": len(df)})
        d = pd.DataFrame(rows)
        out[label] = {"per_seed": rows, "inflation": float(d.inflation.mean()),
                      "inflation_common_support": float(d.inflation_common_support.mean()),
                      "cells_without_new_mean": float(d.cells_without_new.mean()),
                      "images_outside_support_share": float((d.images_outside_support / d.test_images).mean())}
        print(f"Eq.1 {label}: inflation {100 * out[label]['inflation']:.1f} | common support "
              f"{100 * out[label]['inflation_common_support']:.1f} | cells without new {out[label]['cells_without_new_mean']:.1f} "
              f"| images outside support {100 * out[label]['images_outside_support_share']:.1f}%")
    return out


def budget_caps(n_draws):
    out = {}
    for model, prefix in (("A", ""), ("B", "pandermfrozen_")):
        units = [(s, f"{prefix}fedavg_r9_loco{h}_s{s}") for s in (0, 1, 2) for h in range(6)]
        us = [E.Unit(E.load("runs", os.path.join("splits", "e1"), s, r)["test"], "lac") for s, r in units]
        avail = np.mean([[nl // 2 for _, _, nl in u.by_center] for u in us], 0)
        out[model] = {"available_calibration_lesions_by_client": avail.tolist(), "caps": {}}
        rng = np.random.default_rng(20261109)
        for cap in (10, 25, 50, 100):
            argmin = {r: np.zeros(6) for r in ("clientwise", "eb", "global")}
            for u in us:
                cov = {r: [] for r in argmin}
                for _ in range(n_draws):
                    cal_s, n_c, ev = [], [], []
                    for idx, inv, nl in u.by_center:
                        perm = rng.permutation(nl)
                        half = nl // 2
                        cl, el = perm[: min(cap, half)], perm[half:]
                        cal_s.append(u.s[idx[np.isin(inv, cl)]])
                        n_c.append(len(cl))
                        ev.append(idx[np.isin(inv, el)])
                    q = np.array([E.level_quantile(s, math.ceil((len(s) + 1) * (1 - ALPHA)) / len(s)) for s in cal_s])
                    qv = [E.eb_inputs(s, ALPHA, len(s), n) for s, n in zip(cal_s, n_c)]
                    allc = np.concatenate(cal_s)
                    qg = E.level_quantile(allc, math.ceil((len(allc) + 1) * (1 - ALPHA)) / len(allc))
                    thr = {"clientwise": q, "eb": E.eb_thresholds(np.array([x[0] for x in qv]), np.array([x[1] for x in qv])),
                           "global": np.full(6, qg)}
                    for r, t in thr.items():
                        cov[r].append([np.mean(u.s[e] <= t[c]) for c, e in enumerate(ev)])
                for r in argmin:
                    argmin[r][int(np.argmin(np.percentile(np.array(cov[r]), 10, 0)))] += 1
            out[model]["caps"][cap] = {"realized_calibration_lesions_by_client": np.minimum(avail, cap).tolist(),
                                       "worst_client_share": {r: (v / len(us)).tolist() for r, v in argmin.items()}}
        print(f"budget {model}: available lesions by client {np.round(avail).astype(int).tolist()}; worst client (EB) at cap 100: "
              f"{np.round(out[model]['caps'][100]['worst_client_share']['eb'], 2).tolist()}")
    return out


def e8_seeds():
    out = {}
    for run in ("fedavg_r9", "effb0frozen_fedavg_r9", "pandermfrozen_fedavg_r9", "local_e20", "pandermfrozen_fedper_r9"):
        d = E8.load_family("runs", os.path.join("splits", "e8"), run, [0, 1, 2])
        out[run] = {str(s): {"ba_effect": E8.metrics(d[s], np.ones(len(d[s]["y"])))["ba"], "n_targets": int(len(d[s]["y"])),
                             "sanity_nontarget": E8.sanity(d[s])} for s in d}
        print(f"E8 {run}: seed effects " + " ".join(f"{100 * v['ba_effect']:+.1f}" for v in out[run].values()))
    sib = []
    for s in (0, 1, 2):
        man = pd.read_csv(os.path.join("splits", "e8", f"sibexp_manifest_seed{s}.csv"))
        col = [c for c in man.columns if "sib" in c.lower()]
        if col:
            v = man[col[0]].to_numpy()
            sib.append({"seed": s, "column": col[0], "median": float(np.median(v)), "mean": float(v.mean()),
                        "p90": float(np.percentile(v, 90)), "max": int(v.max())})
    out["siblings_per_target"] = sib
    print("siblings per target:", sib)
    return out


def main():
    res = {"loco_heldout": loco_heldout(int(os.environ.get("FU_DRAWS", 1000))), "eq1_support": eq1_support(),
           "budget_caps": budget_caps(int(os.environ.get("FU_BUDGET_DRAWS", 300))), "e8_seeds": e8_seeds()}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(os.path.join("results", "Followup_analyses.json"), "w"), indent=1, default=float)
    print("saved results/Followup_analyses.json")


if __name__ == "__main__":
    main()
