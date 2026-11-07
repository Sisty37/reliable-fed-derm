"""
Data and split audit for the supplement (post hoc reporting, protocol v0.12, Section 12, 7 Nov 2026).
No training. Uses the split files, the official FLamby split and saved run metrics.

  S-A  provenance per client: source name, institution, images, lesions, multiplicity distribution, lesion-ID
       origin and validity checks (missing IDs, IDs equal to image names, label conflicts, lesions in 2 clients);
  S-B  per client x class: images and lesions (all data) and mean train / val / test images over seeds for both
       split types;
  S-C  leakage per client: official FLamby split and image-level re-splits; test images whose lesion has training
       images in the same client, only in another client, or only in validation;
  S-D  dose-response sensitivity: WLS slope of per-client inflation on leakage rate with each client left out;
  S-E  seed-level pooled balanced accuracy of every E2/E7 run (models A and B) and paired differences to FedAvg;
  S-F  realized train / val / test sizes per split and seed.

Usage (from the repo folder):  python scripts/audit_supplement.py
Output: results/Audit_supplement.json, tables/TableS_*.tex
"""
import glob
import json
import os

import numpy as np
import pandas as pd

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
SOURCES = {0: ("BCN20000", "Hospital Clínic de Barcelona", "ISIC 2019 metadata"),
           1: ("HAM vidir_molemax", "ViDIR Group, Medical University of Vienna (MoleMax device)", "HAM10000 metadata"),
           2: ("HAM vidir_modern", "ViDIR Group, Medical University of Vienna (modern dermatoscopes)", "HAM10000 metadata"),
           3: ("HAM rosendahl", "Skin cancer practice of C. Rosendahl, Queensland", "HAM10000 metadata"),
           4: ("MSK", "Memorial Sloan Kettering Cancer Center", "ISIC 2019 metadata"),
           5: ("HAM vienna_dias", "ViDIR Group, Medical University of Vienna (DIAS)", "HAM10000 metadata")}
SEEDS = range(5)


def wls_slope(x, y, w):
    X = np.c_[np.ones(len(x)), x]
    W = np.diag(w)
    return float(np.linalg.solve(X.T @ W @ X, X.T @ W @ y)[1])


def main():
    res = {}
    off = pd.read_csv(os.path.join("splits", "split_official.csv"))
    # ---------- S-A provenance and ID checks
    rows = []
    for c, g in off.groupby("center"):
        per = g.groupby("lesion_id").size()
        rows.append({"client": c, "source": SOURCES[c][0], "institution": SOURCES[c][1], "lesion_id_from": SOURCES[c][2],
                     "images": len(g), "lesions": per.size, "img_per_lesion": round(len(g) / per.size, 2),
                     "lesions_2plus": int((per >= 2).sum()), "median": float(per.median()),
                     "p90": float(per.quantile(0.9)), "max": int(per.max())})
    span = off.groupby("lesion_id").center.nunique()
    res["provenance"] = rows
    res["id_checks"] = {"images": len(off), "lesions": int(off.lesion_id.nunique()),
                        "missing_ids": int(off.lesion_id.isna().sum()),
                        "ids_equal_image_name": int((off.lesion_id.astype(str).str.replace("IMG_", "") == off.image).sum()),
                        "lesions_with_conflicting_labels": int((off.groupby("lesion_id").target.nunique() > 1).sum()),
                        "lesions_in_two_clients": int((span > 1).sum()),
                        "clients_of_shared_lesions": sorted({tuple(sorted(v)) for v in
                                                             off[off.lesion_id.isin(span[span > 1].index)]
                                                             .groupby("lesion_id").center.unique()}),
                        "sum_of_client_lesion_counts": int(sum(r["lesions"] for r in rows))}

    # ---------- S-B per client x class counts
    cc = off.groupby(["center", "target"]).agg(images=("image", "size"), lesions=("lesion_id", "nunique")).reset_index()
    res["client_class_counts"] = cc.to_dict("records")
    split_counts = {}
    for kind in ("image", "lesion"):
        acc = []
        for s in SEEDS:
            d = pd.read_csv(os.path.join("splits", "e1", f"resplit_{kind}_seed{s}_val.csv"))
            acc.append(d.groupby(["center", "target", "fold"]).size().unstack("fold").fillna(0))
        split_counts[kind] = (sum(acc) / len(acc)).round(1)
    res["split_counts_mean"] = {k: v.reset_index().to_dict("records") for k, v in split_counts.items()}

    # ---------- S-C leakage
    def leak_table(d):
        tr, va, te = d[d.fold == "train"], d[d.fold == "val"], d[d.fold == "test"]
        out = []
        for c in range(6):
            t = te[te.center == c]
            same = set(tr.lesion_id[tr.center == c])
            other = set(tr.lesion_id[tr.center != c])
            vl = set(va.lesion_id) - set(tr.lesion_id)
            out.append({"client": c, "test_images": len(t),
                        "leaked_same_client": float(t.lesion_id.isin(same).mean()),
                        "leaked_other_client_only": float((t.lesion_id.isin(other) & ~t.lesion_id.isin(same)).mean()),
                        "validation_only": float(t.lesion_id.isin(vl).mean())})
        tot = {"client": "all", "test_images": len(te),
               "leaked_same_client": float(np.sum([r["leaked_same_client"] * r["test_images"] for r in out]) / len(te)),
               "leaked_other_client_only": float(np.sum([r["leaked_other_client_only"] * r["test_images"] for r in out]) / len(te)),
               "validation_only": float(np.sum([r["validation_only"] * r["test_images"] for r in out]) / len(te))}
        return out + [tot]
    res["leakage_official"] = leak_table(off.assign(fold=off.fold))
    per_seed = [leak_table(pd.read_csv(os.path.join("splits", "e1", f"resplit_image_seed{s}_val.csv"))) for s in SEEDS]
    res["leakage_image_resplits_mean"] = [{k: (float(np.mean([p[i][k] for p in per_seed])) if k != "client" else per_seed[0][i][k])
                                          for k in per_seed[0][i]} for i in range(7)]
    res["leakage_lesion_resplits_max"] = float(max(
        leak_table(pd.read_csv(os.path.join("splits", "e1", f"resplit_lesion_seed{s}_val.csv")))[-1]["leaked_same_client"]
        + leak_table(pd.read_csv(os.path.join("splits", "e1", f"resplit_lesion_seed{s}_val.csv")))[-1]["leaked_other_client_only"]
        for s in SEEDS))

    # ---------- S-D dose-response leave-one-client-out
    dr = {}
    for model, f in (("A", "E7_analysis.json"), ("B", "E7_analysis_panderm.json")):
        p = json.load(open(os.path.join("results", f)))["point"]["per_center"]
        lr, w = np.array(p["leakage_rate"]), np.array(p["new_lesion_images"])
        dr[model] = {}
        for m, inf in p["inflation"].items():
            y = 100 * np.array(inf)
            full = wls_slope(lr, y, w) / 100
            loo = {c: wls_slope(np.delete(lr, c), np.delete(y, c), np.delete(w, c)) / 100 for c in range(6)}
            dr[model][m] = {"slope_all": full, "slope_leave_one_out": loo,
                            "range_leave_one_out": [min(loo.values()), max(loo.values())]}
    res["dose_response_loo"] = dr

    # ---------- S-E seed-level BA
    seed_ba = {}
    for kind in ("image", "lesion"):
        for f in sorted(glob.glob(os.path.join("runs", f"resplit_{kind}_seed*_val", "*", "metrics.json"))):
            run = os.path.basename(os.path.dirname(f))
            if "loco" in run or "r27" in run or "effb0" in run:
                continue
            m = json.load(open(f))
            base = run.rsplit("_s", 1)[0]
            seed_ba.setdefault(f"{kind}|{base}", {})[int(run.rsplit("_s", 1)[1])] = m["bacc_pooled"]
    res["seed_level_pooled_ba"] = seed_ba
    diffs = {}
    for key, v in seed_ba.items():
        kind, base = key.split("|")
        ref = f"{kind}|{'pandermfrozen_' if base.startswith('pandermfrozen_') else ''}fedavg_r9"
        if key == ref or ref not in seed_ba:
            continue
        common = sorted(set(v) & set(seed_ba[ref]))
        d = [v[s] - seed_ba[ref][s] for s in common]
        diffs[key] = {"mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)) if len(d) > 1 else None,
                      "min": float(min(d)), "max": float(max(d)), "n_seeds": len(d)}
    res["paired_ba_minus_fedavg"] = diffs

    # ---------- S-F realized sizes
    sizes = []
    for kind in ("image", "lesion"):
        for s in SEEDS:
            d = pd.read_csv(os.path.join("splits", "e1", f"resplit_{kind}_seed{s}_val.csv"))
            sizes.append({"split": kind, "seed": s, **{f"{f}_images": int((d.fold == f).sum()) for f in ("train", "val", "test")},
                          **{f"{f}_lesions": int(d.lesion_id[d.fold == f].nunique()) for f in ("train", "val", "test")}})
    res["realized_sizes"] = sizes

    os.makedirs("results", exist_ok=True)
    json.dump(res, open(os.path.join("results", "Audit_supplement.json"), "w"), indent=1, default=str)

    # ---------- printout
    print(pd.DataFrame(rows).drop(columns="institution").to_string(index=False))
    print("ID checks:", res["id_checks"])
    print("\nLeakage, official split:")
    print(pd.DataFrame(res["leakage_official"]).round(3).to_string(index=False))
    print("\nLeakage, image-level re-splits (mean of 5 seeds):")
    print(pd.DataFrame(res["leakage_image_resplits_mean"]).round(3).to_string(index=False))
    print("max leakage in lesion-grouped re-splits:", res["leakage_lesion_resplits_max"])
    print("\nDose-response slope (pts per 100% leakage), all clients and leave-one-out range:")
    for model, v in dr.items():
        for m, x in v.items():
            print(f"  {model} {m:14s} {100 * x['slope_all']:6.1f}  LOO {100 * x['range_leave_one_out'][0]:6.1f} .. "
                  f"{100 * x['range_leave_one_out'][1]:6.1f}  without client 0: {100 * x['slope_leave_one_out'][0]:6.1f}")
    print("\nPaired pooled-BA difference to FedAvg (pts, mean / SD / range over seeds):")
    for k, v in sorted(diffs.items()):
        print(f"  {k:40s} {100 * v['mean']:+6.1f}  SD {100 * (v['sd'] or 0):4.1f}  [{100 * v['min']:+.1f}, {100 * v['max']:+.1f}]  n={v['n_seeds']}")
    print(pd.DataFrame(sizes).to_string(index=False))


if __name__ == "__main__":
    main()
