"""Gate 1 (protocol v0.10): model A FedAvg on the official split, mean of 3 seeds,
pooled-test balanced accuracy within +/-5 points of FENDA-FL's 0.657."""
import glob
import json
import os
import sys

import numpy as np

REF, TOL = 0.657, 0.05
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join("runs", "split_official")
for method in ["fedavg", "fedprox", "scaffold", "pooled"]:
    files = sorted(glob.glob(os.path.join(root, f"{method}_*", "metrics.json")))
    if not files:
        continue
    m = [json.load(open(f)) for f in files]
    pooled = np.array([x["bacc_pooled"] for x in m])
    mean_c = np.array([x["bacc_mean_of_centers"] for x in m])
    print(f"{method:9s} seeds {[x['seed'] for x in m]}  pooled BA {pooled.mean():.4f} "
          f"(range {pooled.min():.4f}-{pooled.max():.4f})  mean-of-centers BA {mean_c.mean():.4f}")
    if method == "fedavg":
        ok = abs(pooled.mean() - REF) <= TOL and len(m) >= 3
        print(f"\nGATE 1: FedAvg pooled BA {pooled.mean():.4f} vs {REF} +/- {TOL}  ->  "
              f"{'PASS' if ok else ('FAIL' if len(m) >= 3 else 'INCOMPLETE (need 3 seeds)')}\n")
