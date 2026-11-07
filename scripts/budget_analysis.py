"""
Post hoc training-budget controls (review points 50-51; descriptive, not a pre-specified test).

For each model (A, A-frozen, B) and each training variant, on seeds 0-2:
  * apparent inflation in clients 0, 2 and 3 (the H3b statistic of E2/E7, Eq. 1 per client, weighted by
    new-lesion counts) on the image-level splits, with a 95% lesion-bootstrap interval (1,000 replicates);
  * pooled balanced accuracy on image-level and lesion-grouped splits and their difference.
Variants: FedAvg-FT with 1, 3 (protocol), 10, 30 (and 100 for frozen encoders) local steps; FedAvg with 9
(protocol) and 27 rounds; pooled and local-only with 5, 20 (protocol) and 60 epochs.
The question is whether the ordering of inflation across methods (FedAvg < FedAvg-FT, pooled < local-only)
depends on the training budget.

Usage (from the repo folder, after scripts\\run_budget_ft.ps1):  python scripts\\budget_analysis.py
Output: results/Budget_controls.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e2_analysis as E2  # noqa: E402

SEEDS = [0, 1, 2]
VARIANTS = {
    "A": ("", ["fedavg_r9", "fedavgft1_r9", "fedavgft_r9", "fedavgft10_r9", "fedavgft30_r9", "pooled_e20", "local_e20"]),
    "A-frozen": ("effb0frozen_", ["fedavg_r9", "fedavg_r27", "fedavgft1_r9", "fedavgft_r9", "fedavgft10_r9",
                                  "fedavgft30_r9", "fedavgft100_r9", "pooled_e5", "pooled_e20", "pooled_e60",
                                  "local_e5", "local_e20", "local_e60"]),
    "B": ("pandermfrozen_", ["fedavg_r9", "fedavg_r27", "fedavgft1_r9", "fedavgft_r9", "fedavgft10_r9",
                             "fedavgft30_r9", "fedavgft100_r9", "pooled_e5", "pooled_e20", "pooled_e60",
                             "local_e5", "local_e20", "local_e60"]),
}
PROTOCOL = {"fedavg_r9", "fedavgft_r9", "pooled_e20", "local_e20"}


def available(prefix, methods):
    ok = []
    for m in methods:
        if all(os.path.exists(os.path.join("runs", f"resplit_{k}_seed{s}_val", f"{prefix}{m}_s{s}", "preds.npz"))
               for k in ("image", "lesion") for s in SEEDS):
            ok.append(m)
        else:
            print(f"  missing runs for {prefix}{m}: skipped")
    return ok


def h3b(data, w):
    """Inflation per method in clients 0, 2, 3, weighted by new-lesion counts (seed mean)."""
    infl, cnt = [], []
    for s in sorted(data):
        row, c_row = [], []
        for c in E2.H3B_CENTERS:
            v, n = E2.center_inflation(data[s], w[s], c)
            row.append(v)
            c_row.append(n)
        infl.append(row)
        cnt.append(c_row)
    mi = np.nanmean(np.array(infl), 0)                  # clients x methods
    wts = np.array(cnt).mean(0)
    return np.array([np.average(mi[:, j], weights=wts) for j in range(mi.shape[1])])


def main():
    n_boot = int(os.environ.get("N_BOOT", 1000))
    res = {}
    for model, (prefix, methods) in VARIANTS.items():
        methods = available(prefix, methods)
        if not methods:
            continue
        E2.METHODS[:] = methods
        data = {k: {s: E2.load("runs", os.path.join("splits", "e1"), k, s, prefix) for s in SEEDS} for k in ("image", "lesion")}
        ones = {s: np.ones(len(d["y"])) for s, d in data["image"].items()}
        point = h3b(data["image"], ones)
        ba = {k: np.mean([E2.ba(d, np.ones(len(d["y"]))) for d in data[k].values()], 0) for k in data}
        rng = np.random.default_rng(20261108)
        boots = []
        for _ in range(n_boot):
            w = {s: rng.multinomial(d["n_les"], np.full(d["n_les"], 1 / d["n_les"]))[d["inv"]].astype(float)
                 for s, d in data["image"].items()}
            boots.append(h3b(data["image"], w))
        boots = np.array(boots)
        j0 = methods.index("fedavg_r9")
        res[model] = {}
        print(f"\n=== model {model}: seeds {SEEDS}; inflation in clients 0, 2, 3 (pts) and pooled BA ===")
        for j, m in enumerate(methods):
            lo, hi = np.nanpercentile(boots[:, j], [2.5, 97.5])
            dlo, dhi = np.nanpercentile(boots[:, j] - boots[:, j0], [2.5, 97.5])
            res[model][m] = {"inflation": float(point[j]), "ci": [float(lo), float(hi)],
                             "minus_fedavg_r9": float(point[j] - point[j0]), "minus_fedavg_r9_ci": [float(dlo), float(dhi)],
                             "ba_image": float(ba["image"][j]), "ba_lesion": float(ba["lesion"][j]),
                             "protocol_setting": m in PROTOCOL}
            print(f"  {m:15s}{'*' if m in PROTOCOL else ' '} inflation {100 * point[j]:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"
                  f"  minus FedAvg {100 * (point[j] - point[j0]):+5.1f} [{100 * dlo:+5.1f}, {100 * dhi:+5.1f}]"
                  f"  | BA image {ba['image'][j]:.3f} lesion {ba['lesion'][j]:.3f}")
    print("\n* = protocol setting")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(os.path.join("results", "Budget_controls.json"), "w"), indent=2)
    print("saved results/Budget_controls.json")


if __name__ == "__main__":
    main()
