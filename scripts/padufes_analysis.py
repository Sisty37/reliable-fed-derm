"""
E6 PAD-UFES-20 domain-shift stress test, analysis (protocol v0.11, Sections 2 and 5;
choices fixed in results/E6_padufes_implementation_notes.md before predictions were made).

Models: lesion-grouped FedAvg models A and B, seeds 0-4 (the E5 models), predictions from
scripts/padufes_predict.py. Probabilities: softmax with the model's global temperature (Fed-ISIC2019
validation lesions, as in E5), restricted to the six shared classes (MEL, NV, BCC, AK, BKL, SCC) and
renormalized. Reference: the same model on its Fed-ISIC2019 lesion-grouped test images of those six
classes, with the same restriction.

Accuracy / calibration: balanced accuracy (6 classes), top-label ECE (15 bins), Brier score.
Conformal (LAC primary, APS; alpha 0.05, 0.1, 0.2), threshold level ceil((m+1)(1-alpha))/m:
  (i)  transfer: calibration scores = all Fed-ISIC2019 lesion-grouped test images of the six classes
       (all centers pooled); coverage and set size on all PAD-UFES-20 images;
  (ii) recalibrated: PAD-UFES-20 patients split 50/50 at random, 100 times; threshold from the
       calibration half, coverage and set size on the other half (mean and 10th percentile over splits).
Coverage is also reported per class (class-conditional). Intervals: patient bootstrap of PAD-UFES-20
(1,000 replicates, seeds averaged inside each replicate) for BA, ECE and transfer coverage.

Usage (from the repo folder):  python scripts\\padufes_analysis.py
Writes results/E6_padufes.json and prints the tables.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e3_e6_analysis import bacc, ece, load  # noqa: E402
from e5_conformal import softmax  # noqa: E402

SHARED = [0, 1, 2, 3, 4, 7]
NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "SCC"]
ALPHAS = (0.05, 0.1, 0.2)
MODELS = {"A": "", "B": "pandermfrozen_"}


def restrict(p, y):
    """Six shared classes, renormalized; labels re-indexed 0-5 (images of other classes dropped)."""
    keep = np.isin(y, SHARED)
    q = p[keep][:, SHARED]
    q = q / q.sum(1, keepdims=True)
    yy = np.searchsorted(SHARED, y[keep])
    return q, yy, keep


def brier6(p, y):
    return float(np.mean(((p - np.eye(len(SHARED))[y]) ** 2).sum(1)))


def score_all(p, kind):
    if kind == "lac":
        return 1.0 - p
    order = np.argsort(-p, 1)
    out = np.empty_like(p)
    np.put_along_axis(out, order, np.cumsum(np.take_along_axis(p, order, 1), 1), 1)
    return out


def threshold(s, alpha):
    m = len(s)
    level = math.ceil((m + 1) * (1 - alpha)) / m
    return 1.0 if level > 1 else float(np.sort(s)[max(int(math.ceil(level * m)) - 1, 0)])


def cover(S, y, t, w=None):
    """Weighted coverage, mean set size, and coverage per class."""
    w = np.ones(len(y)) if w is None else w
    inset = S <= t
    hit = inset[np.arange(len(y)), y]
    cls = [float(np.average(hit[y == k], weights=w[y == k])) if w[y == k].sum() > 0 else np.nan
           for k in range(len(SHARED))]
    return float(np.average(hit, weights=w)), float(np.average(inset.sum(1), weights=w)), cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split-dir", default=os.path.join("splits", "e1"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join("results", "E6_padufes.json"))
    a = ap.parse_args()
    res = {}
    for mname, prefix in MODELS.items():
        rng = np.random.default_rng(20261106)
        per = []
        for s in a.seeds:
            iso = load(a.runs, a.split_dir, "lesion", s, f"{prefix}fedavg_r9_s{s}")
            z = np.load(os.path.join(a.runs, "padufes20", f"{prefix}fedavg_r9_s{s}", "preds.npz"))
            pi, yi, _ = restrict(iso["p"], iso["y"])
            pp, yp, keep = restrict(softmax(z["logits"].astype(np.float64) / iso["T"]), z["target"].astype(int))
            assert keep.all()
            per.append({"pi": pi, "yi": yi, "pp": pp, "yp": yp, "T": iso["T"],
                        "patient": z["patient"].astype(str)})
        patients, pinv = np.unique(per[0]["patient"], return_inverse=True)
        assert all((d["patient"] == per[0]["patient"]).all() for d in per)

        r = {"temperature": [d["T"] for d in per], "n_pad": len(per[0]["yp"]), "n_patients": len(patients)}
        # accuracy and calibration
        r["isic_ba"] = float(np.mean([bacc(d["pi"], d["yi"]) for d in per]))
        r["pad_ba"] = float(np.mean([bacc(d["pp"], d["yp"]) for d in per]))
        r["isic_ece"] = float(np.mean([ece(d["pi"], d["yi"]) for d in per]))
        r["pad_ece"] = float(np.mean([ece(d["pp"], d["yp"]) for d in per]))
        r["isic_brier"] = float(np.mean([brier6(d["pi"], d["yi"]) for d in per]))
        r["pad_brier"] = float(np.mean([brier6(d["pp"], d["yp"]) for d in per]))
        r["pad_recall_by_class"] = np.mean([[np.mean(d["pp"][d["yp"] == k].argmax(1) == k) for k in range(6)]
                                            for d in per], 0).tolist()
        r["pad_confusion_mean"] = np.mean([[np.bincount(d["pp"][d["yp"] == k].argmax(1), minlength=6) for k in range(6)]
                                           for d in per], 0).tolist()

        # conformal
        boot_w = [rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)))[pinv].astype(float)
                  for _ in range(a.n_boot)]
        for kind in ("lac", "aps"):
            for alpha in ALPHAS:
                key = f"{kind}_a{alpha}"
                tr = {"isic_cov": [], "cov": [], "size": [], "cls": [], "boot": np.zeros(a.n_boot)}
                rc = {"cov": [], "q10": [], "size": [], "cls": []}
                for d in per:
                    Si, Sp = score_all(d["pi"], kind), score_all(d["pp"], kind)
                    t = threshold(Si[np.arange(len(d["yi"])), d["yi"]], alpha)
                    ci, _, _ = cover(Si, d["yi"], t)
                    c, sz, cl = cover(Sp, d["yp"], t)
                    tr["isic_cov"].append(ci)
                    tr["cov"].append(c)
                    tr["size"].append(sz)
                    tr["cls"].append(cl)
                    for b, w in enumerate(boot_w):
                        tr["boot"][b] += cover(Sp, d["yp"], t, w)[0] / len(per)
                    srng = np.random.default_rng(1000 + int(alpha * 100) + (0 if kind == "lac" else 7))
                    covs, sizes, cls = [], [], []
                    sp_true = Sp[np.arange(len(d["yp"])), d["yp"]]
                    for _ in range(a.n_splits):
                        cal_pat = srng.random(len(patients)) < 0.5
                        cal = cal_pat[pinv]
                        t2 = threshold(sp_true[cal], alpha)
                        c2, s2, cl2 = cover(Sp[~cal], d["yp"][~cal], t2)
                        covs.append(c2)
                        sizes.append(s2)
                        cls.append(cl2)
                    rc["cov"].append(np.mean(covs))
                    rc["q10"].append(np.percentile(covs, 10))
                    rc["size"].append(np.mean(sizes))
                    rc["cls"].append(np.nanmean(cls, 0))
                r[key] = {"transfer": {"isic_coverage": float(np.mean(tr["isic_cov"])), "coverage": float(np.mean(tr["cov"])),
                                       "coverage_ci95": [float(np.percentile(tr["boot"], 2.5)), float(np.percentile(tr["boot"], 97.5))],
                                       "set_size": float(np.mean(tr["size"])), "coverage_by_class": np.mean(tr["cls"], 0).tolist()},
                          "recalibrated": {"coverage": float(np.mean(rc["cov"])), "coverage_q10": float(np.mean(rc["q10"])),
                                           "set_size": float(np.mean(rc["size"])), "coverage_by_class": np.mean(rc["cls"], 0).tolist()}}
        # bootstrap intervals for BA and ECE
        bb, be = [], []
        for w in boot_w[:a.n_boot]:
            bb.append(np.mean([np.mean([np.average(d["pp"][d["yp"] == k].argmax(1) == k, weights=w[d["yp"] == k])
                                        for k in range(6) if w[d["yp"] == k].sum() > 0]) for d in per]))
            be.append(np.mean([ece(d["pp"], d["yp"], w) for d in per]))
        r["pad_ba_ci95"] = [float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))]
        r["pad_ece_ci95"] = [float(np.percentile(be, 2.5)), float(np.percentile(be, 97.5))]
        res[mname] = r

        print(f"\n=== Model {mname}: Fed-ISIC2019 lesion-grouped test vs PAD-UFES-20 (6 classes; mean of {len(per)} seeds) ===")
        print(f"  PAD-UFES-20: {r['n_pad']} images, {r['n_patients']} patients | global T = "
              + " ".join(f"{t:.2f}" for t in r["temperature"]))
        print(f"  balanced accuracy  Fed-ISIC {r['isic_ba']:.3f} | PAD {r['pad_ba']:.3f} "
              f"[{r['pad_ba_ci95'][0]:.3f}, {r['pad_ba_ci95'][1]:.3f}]")
        print(f"  ECE                Fed-ISIC {r['isic_ece']:.3f} | PAD {r['pad_ece']:.3f} "
              f"[{r['pad_ece_ci95'][0]:.3f}, {r['pad_ece_ci95'][1]:.3f}]")
        print(f"  Brier              Fed-ISIC {r['isic_brier']:.3f} | PAD {r['pad_brier']:.3f}")
        print("  PAD recall by class " + " ".join(f"{n} {v:.2f}" for n, v in zip(NAMES, r["pad_recall_by_class"])))
        print(f"  {'score':5s} {'alpha':5s} | transfer: ISIC cov  PAD cov [95% CI]        size | recalibrated: cov  q10   size")
        for kind in ("lac", "aps"):
            for alpha in ALPHAS:
                t, c = r[f"{kind}_a{alpha}"]["transfer"], r[f"{kind}_a{alpha}"]["recalibrated"]
                print(f"  {kind:5s} {alpha:<5} | {t['isic_coverage']:.3f}     {t['coverage']:.3f} "
                      f"[{t['coverage_ci95'][0]:.3f}, {t['coverage_ci95'][1]:.3f}]  {t['set_size']:.2f} | "
                      f"{c['coverage']:.3f}  {c['coverage_q10']:.3f}  {c['set_size']:.2f}")
        t, c = r["lac_a0.1"]["transfer"], r["lac_a0.1"]["recalibrated"]
        print("  LAC a0.1 coverage by class  transfer:     " + " ".join(f"{n} {v:.2f}" for n, v in zip(NAMES, t["coverage_by_class"])))
        print("                              recalibrated: " + " ".join(f"{n} {v:.2f}" for n, v in zip(NAMES, c["coverage_by_class"])))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
