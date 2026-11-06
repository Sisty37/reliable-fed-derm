"""
E8 analysis (protocol v0.12, Section 5c): effect of sibling exposure on fixed target images, and E9 summary.

For every model (method x encoder x seed) the two crossover arms give each target image one prediction
with its siblings in training (exposed) and one without (unexposed). Reported per model family:

  * exposure effect on balanced accuracy of the target images = BA(exposed) - BA(unexposed), over classes
    with >= 5 target images; also accuracy, melanoma recall, NLL and Brier score (all on the same images);
  * the same effect per center for centers with >= 50 target lesions;
  * a sanity check: BA on the non-target test images (never exposed) in arm 1 minus arm 2 (should be ~0);
  * differences in exposure effect between methods (same seeds, same target images).
Intervals: 2,000 paired bootstrap replicates over target lesions (each seed's lesions resampled, seeds
averaged inside each replicate). They reflect test-sample uncertainty given the trained models.

Primary test (pre-specified, v0.12): model A FedAvg; sibling exposure causes evaluation optimism if the
BA effect is >= 2 points with the 95% interval above zero.

E9 (descriptive): model B balanced accuracy and inflation for the capacity-matched two-layer heads
(fedavgmlp, pooledmlp, localmlp) next to the FedPer-style head, via e2_analysis.py-style contrasts.

Usage (from the repo folder):  python scripts/e8_analysis.py [--n-boot 2000]
Output: results/E8_sibling_exposure.json (+ printed tables).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

N_CLASSES, MIN_CLASS, MIN_CENTER = 8, 5, 50
FAMILIES = [("A", "fedavg_r9"), ("A", "fedper_r9"), ("A", "pooled_e20"), ("A", "local_e20"),
            ("A-frozen", "effb0frozen_fedavg_r9"), ("A-frozen", "effb0frozen_pooled_e20"),
            ("A-frozen", "effb0frozen_local_e20"),
            ("B", "pandermfrozen_fedavg_r9"), ("B", "pandermfrozen_fedper_r9"),
            ("B", "pandermfrozen_pooled_e20"), ("B", "pandermfrozen_local_e20")]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_family(runs, split_dir, run, seeds):
    """Per seed: target table with exposed / unexposed probabilities, plus non-target sanity data."""
    out = {}
    for s in seeds:
        man = pd.read_csv(os.path.join(split_dir, f"sibexp_manifest_seed{s}.csv")).set_index("target_image")
        arms = {}
        for arm in (1, 2):
            f = os.path.join(runs, f"sibexp_lesion_seed{s}_arm{arm}", f"{run}_s{s}", "preds.npz")
            if not os.path.exists(f):
                return None
            p = np.load(f)
            arms[arm] = pd.DataFrame({"image": p["image"].astype(str), "y": p["target"].astype(int),
                                      "center": p["center"].astype(int)}).assign(
                prob=list(softmax(p["logits"].astype(np.float64))))
        a1, a2 = (arms[k].set_index("image") for k in (1, 2))
        assert (a1.index == a2.index).all(), "arms must share the same test images in the same order"
        tgt = man.index.to_numpy()
        grp = man.group.to_numpy()
        p1, p2 = np.stack(a1.loc[tgt, "prob"].to_numpy()), np.stack(a2.loc[tgt, "prob"].to_numpy())
        exposed = np.where((grp == 1)[:, None], p1, p2)
        unexposed = np.where((grp == 1)[:, None], p2, p1)
        non = ~a1.index.isin(tgt)
        out[s] = {"y": man.target.to_numpy(), "center": man.center.to_numpy(), "lesion": man.lesion_id.to_numpy(),
                  "exp": exposed, "unexp": unexposed,
                  "non_y": a1.y.to_numpy()[non], "non_p1": np.stack(a1.prob.to_numpy()[non]),
                  "non_p2": np.stack(a2.prob.to_numpy()[non]), "non_center": a1.center.to_numpy()[non]}
    return out


def ba(y, p, w, classes):
    pred = p.argmax(1)
    rec = []
    for k in classes:
        m = (y == k)
        if (w[m]).sum() > 0:
            rec.append(np.sum(w[m] * (pred[m] == k)) / np.sum(w[m]))
    return float(np.mean(rec)) if rec else np.nan


def metrics(d, w, mask=None):
    """Exposed-minus-unexposed effects on target images (weights w = bootstrap multiplicities)."""
    y = d["y"]
    m = np.ones(len(y), bool) if mask is None else mask
    w = w * m
    classes = [k for k in range(N_CLASSES) if (y[m] == k).sum() >= MIN_CLASS]
    res = {}
    for name, p in (("exp", d["exp"]), ("unexp", d["unexp"])):
        py = p[np.arange(len(y)), y]
        res[name] = {"ba": ba(y, p, w, classes),
                     "acc": float(np.sum(w * (p.argmax(1) == y)) / w.sum()),
                     "mel_recall": float(np.sum(w * ((p.argmax(1) == 0) & (y == 0))) / max(np.sum(w * (y == 0)), 1e-9)),
                     "nll": float(np.sum(w * -np.log(py + 1e-12)) / w.sum()),
                     "brier": float(np.sum(w * ((p - np.eye(N_CLASSES)[y]) ** 2).sum(1)) / w.sum())}
    return {k: res["exp"][k] - res["unexp"][k] for k in res["exp"]} | {"levels": res}


def sanity(d):
    y = d["non_y"]
    classes = [k for k in range(N_CLASSES) if (y == k).sum() >= MIN_CLASS]
    w = np.ones(len(y))
    return ba(y, d["non_p1"], w, classes) - ba(y, d["non_p2"], w, classes)


def lesion_weights(d, rng):
    les, inv = np.unique(d["lesion"], return_inverse=True)
    draw = np.bincount(rng.integers(0, len(les), len(les)), minlength=len(les))
    return draw[inv].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--split-dir", default=os.path.join("splits", "e8"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join("results", "E8_sibling_exposure.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(20261108)

    data = {}
    for enc, run in FAMILIES:
        d = load_family(a.runs, a.split_dir, run, a.seeds)
        if d is None:
            print(f"  (skipped, runs missing: {run})")
            continue
        data[run] = (enc, d)
    if not data:
        raise SystemExit("no complete E8 run families found")

    keys = ["ba", "acc", "mel_recall", "nll", "brier"]
    centers = [c for c in range(6)
               if all((v[1][s]["center"] == c).sum() >= MIN_CENTER for v in data.values() for s in a.seeds)]
    point = {r: {k: float(np.mean([metrics(d[s], np.ones(len(d[s]["y"])))[k] for s in a.seeds])) for k in keys}
             for r, (_, d) in data.items()}
    point_center = {r: {c: float(np.mean([metrics(d[s], np.ones(len(d[s]["y"])), d[s]["center"] == c)["ba"]
                                          for s in a.seeds])) for c in centers} for r, (_, d) in data.items()}
    levels = {r: {lv: {k: float(np.mean([metrics(d[s], np.ones(len(d[s]["y"])))["levels"][lv][k] for s in a.seeds]))
                       for k in keys} for lv in ("exp", "unexp")} for r, (_, d) in data.items()}
    san = {r: float(np.mean([sanity(d[s]) for s in a.seeds])) for r, (_, d) in data.items()}

    boot = {r: {k: [] for k in keys} for r in data}
    boot_c = {r: {c: [] for c in centers} for r in data}
    any_run = next(iter(data.values()))[1]
    for b in range(a.n_boot):
        w = {s: lesion_weights(any_run[s], rng) for s in a.seeds}      # same lesions for every model (paired)
        for r, (_, d) in data.items():
            ms = [metrics(d[s], w[s]) for s in a.seeds]
            for k in keys:
                boot[r][k].append(np.mean([m[k] for m in ms]))
            for c in centers:
                boot_c[r][c].append(np.mean([metrics(d[s], w[s], d[s]["center"] == c)["ba"] for s in a.seeds]))
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]

    res = {"seeds": a.seeds, "n_boot": a.n_boot, "centers_reported": centers,
           "n_target_lesions": {s: int(len(any_run[s]["y"])) for s in a.seeds}, "families": {}}
    print(f"\n=== E8: effect of sibling exposure on fixed target images (exposed - unexposed), seeds {a.seeds} ===")
    print(f"{'model':10s} {'run':28s} {'BA (pts)':>22s} {'MEL recall':>11s} {'NLL':>8s} {'Brier':>8s} {'sanity BA':>10s}")
    for r, (enc, _) in data.items():
        e = point[r]
        res["families"][r] = {"encoder": enc, "effect": e, "ci": {k: ci(boot[r][k]) for k in keys},
                              "levels": levels[r], "sanity_nontarget_ba_arm1_minus_arm2": san[r],
                              "per_center_ba_effect": {c: {"effect": point_center[r][c], "ci": ci(boot_c[r][c])}
                                                       for c in centers}}
        lo, hi = ci(boot[r]["ba"])
        print(f"{enc:10s} {r:28s} {100 * e['ba']:+6.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]  "
              f"{100 * e['mel_recall']:+9.1f}  {e['nll']:+8.3f} {e['brier']:+8.3f} {100 * san[r]:+9.1f}")
    print("\nper-center BA effect (pts): " + ", ".join(f"center {c}" for c in centers))
    for r in data:
        print(f"  {r:28s} " + "  ".join(f"{100 * point_center[r][c]:+5.1f} [{100 * ci(boot_c[r][c])[0]:+.1f}, "
                                        f"{100 * ci(boot_c[r][c])[1]:+.1f}]" for c in centers))

    # method contrasts within encoder (paired by seed and target image)
    res["contrasts"] = {}
    for enc in ("A", "A-frozen", "B"):
        runs_e = [r for r, (e, _) in data.items() if e == enc]
        ref = [r for r in runs_e if "fedavg" in r]
        if not ref:
            continue
        for r in runs_e:
            if r == ref[0]:
                continue
            diff = np.array(boot[r]["ba"]) - np.array(boot[ref[0]]["ba"])
            res["contrasts"][f"{r} - {ref[0]}"] = {"effect": point[r]["ba"] - point[ref[0]]["ba"], "ci": ci(diff)}
    print("\nexposure effect relative to FedAvg of the same encoder (BA pts):")
    for k, v in res["contrasts"].items():
        print(f"  {k:55s} {100 * v['effect']:+5.1f} [{100 * v['ci'][0]:+.1f}, {100 * v['ci'][1]:+.1f}]")

    if "fedavg_r9" in point:
        e, (lo, hi) = point["fedavg_r9"]["ba"], ci(boot["fedavg_r9"]["ba"])
        ok = e >= 0.02 and lo > 0
        res["primary"] = {"run": "fedavg_r9", "effect": e, "ci": [lo, hi], "met": bool(ok),
                          "rule": "BA effect >= 2 points and 95% interval above zero (v0.12)"}
        print(f"\nPRIMARY (v0.12): model A FedAvg exposure effect {100 * e:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}] "
              f"-> {'MET' if ok else 'NOT MET'}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
