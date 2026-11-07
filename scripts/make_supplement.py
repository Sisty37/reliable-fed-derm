"""
Supplementary material (S1-S12) as a stand-alone LaTeX document, built only from files in results/,
splits/ and runs/ (no hand-typed numbers except the protocol chronology, which is taken from the git tags).

Usage (from the repo folder):  python scripts/make_supplement.py
Writes paper/supplement.tex (compile with pdflatex; figure S1 is read from figures/).
"""
import json
import os

import numpy as np

R = "results"
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NAMES = {"fedavg_r9": "FedAvg", "fedprox_r9": "FedProx", "scaffold_r9": "SCAFFOLD", "fedbn_r9": "FedBN",
         "fedavgft_r9": "FedAvg-FT", "fedper_r9": "FedPer", "pooled_e20": "Pooled", "local_e20": "Local-only",
         "fedavgmlp_r9": "FedAvg (2-layer)", "pooledmlp_e20": "Pooled (2-layer)", "localmlp_e20": "Local-only (2-layer)",
         "effb0frozen_fedavg_r9": "FedAvg", "effb0frozen_pooled_e20": "Pooled", "effb0frozen_local_e20": "Local-only",
         "pandermfrozen_fedavg_r9": "FedAvg", "pandermfrozen_fedper_r9": "FedPer", "pandermfrozen_pooled_e20": "Pooled",
         "pandermfrozen_local_e20": "Local-only"}
CONF = {"global": "Global", "clientwise": "Client-wise", "weighted": "Sample-weighted", "fcp": "FCP",
        "shrink": "ShrinkFCP", "eb": "EB-ShrinkFCP", "lsw": "Label-shift-weighted"}


def load(n):
    p = os.path.join(R, n)
    return json.load(open(p)) if os.path.exists(p) else None


def f1(x):
    return f"{x:.1f}".replace("-", "$-$")


def f3(x):
    return f"{x:.3f}".replace("-", "$-$")


def p1(x, ci=None):
    s = f"{100 * x:+.1f}"
    if ci is not None:
        s += f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
    return s.replace("-", "$-$")


def table(caption, label, cols, header, rows, size="\\footnotesize", wide=False):
    env = "table*" if wide else "table"
    return (f"\\begin{{{env}}}[!htbp]\n\\centering\n{size}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            + "\n".join(r + " \\\\" if r != "\\midrule" else r for r in rows)
            + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def s1():
    a = load("Audit_supplement.json")
    out = ["\\section*{S1. Data and split audit}\n",
           "Lesion identifiers come from the HAM10000 metadata (clients 1, 2, 3, 5) and the ISIC 2019 metadata "
           "(clients 0, 4). Checks on all 23,247 images: "
           f"missing identifiers {a['id_checks']['missing_ids']}, identifiers equal to the image name "
           f"{a['id_checks']['ids_equal_image_name']}, lesions with more than one diagnosis "
           f"{a['id_checks']['lesions_with_conflicting_labels']}, lesions with images in two clients "
           f"{a['id_checks']['lesions_in_two_clients']} (all in clients 1 and 2, both from the Vienna institution), "
           f"sum of per-client lesion counts {a['id_checks']['sum_of_client_lesion_counts']:,} "
           f"versus {a['id_checks']['lesions']:,} distinct lesions. Patient identifiers are not available.\n"]
    rows = [f"{r['client']} & {r['source'].replace('_', chr(92) + '_')} & {r['lesion_id_from']} & {r['images']:,} & {r['lesions']:,} & "
            f"{r['img_per_lesion']:.2f} & {r['lesions_2plus']:,} & {r['median']:.0f} & {r['p90']:.0f} & {r['max']}"
            for r in a["provenance"]]
    out.append(table("Clients, sources and lesion multiplicity (images per lesion).", "tab:s1prov", "llp{2.6cm}rrrrrrr",
                     "Client & Source & Lesion IDs from & Images & Lesions & Mean & $\\ge2$ img. & Median & P90 & Max", rows, wide=True))
    cc = {(r["center"], r["target"]): r for r in a["client_class_counts"]}
    rows = []
    for c in range(6):
        rows.append(f"{c} & " + " & ".join(f"{cc[(c, k)]['images']}/{cc[(c, k)]['lesions']}" if (c, k) in cc else "--"
                                         for k in range(8)))
    out.append(table("Images/lesions per client and class (all data).", "tab:s1cls", "l" + "r" * 8,
                     "Client & " + " & ".join(CLASSES), rows, wide=True))
    for kind in ("image", "lesion"):
        sc = a["split_counts_mean"][kind]
        rows = []
        for c in range(6):
            cells = []
            for k in range(8):
                r = [x for x in sc if x["center"] == c and x["target"] == k]
                cells.append("--" if not r else f"{r[0].get('train', 0):.0f}/{r[0].get('val', 0):.0f}/{r[0].get('test', 0):.0f}")
            rows.append(f"{c} & " + " & ".join(cells))
        out.append(table(f"{kind.capitalize()}-level re-splits: mean train/validation/test images per client and class "
                         "(five seeds).", f"tab:s1split{kind}", "l" + "r" * 8, "Client & " + " & ".join(CLASSES), rows,
                         size="\\scriptsize", wide=True))
    rows = [f"{r['split']} & {r['seed']} & {r['train_images']:,} & {r['val_images']:,} & {r['test_images']:,} & "
            f"{r['train_lesions']:,} & {r['val_lesions']:,} & {r['test_lesions']:,}" for r in a["realized_sizes"]]
    out.append(table("Realized split sizes.", "tab:s1sizes", "lrrrrrrr",
                     "Split & Seed & Train img. & Val img. & Test img. & Train les. & Val les. & Test les.", rows))
    rows = []
    for name, tab in (("Official", a["leakage_official"]), ("Re-splits", a["leakage_image_resplits_mean"])):
        for r in tab:
            rows.append(f"{name if r['client'] in (0, '0') else ''} & {r['client']} & {r['test_images']:.0f} & "
                        f"{100 * r['leaked_same_client']:.1f} & {100 * r['leaked_other_client_only']:.1f} & {100 * r['validation_only']:.1f}")
        rows.append("\\midrule")
    rows = rows[:-1]
    out.append(table("Leakage of image-level test images (\\%): lesion with training images in the same client, only in "
                     "another client, or only in the validation fold (temperature scaling only). Lesion-grouped "
                     f"re-splits: maximum leakage {100 * a['leakage_lesion_resplits_max']:.1f}\\%.", "tab:s1leak", "llrrrr",
                     "Split & Client & Test img. & Same client & Other client only & Validation only", rows))
    return "".join(out)


def s3():
    out = ["\\section*{S3. Apparent inflation per client and dose--response sensitivity}\n"]
    a = load("Audit_supplement.json")
    for model, f in (("A", "E7_analysis.json"), ("B", "E7_analysis_panderm.json")):
        d = load(f)["point"]["per_center"]
        rows = [f"{NAMES[m]} & " + " & ".join(f1(100 * x) for x in v) for m, v in d["inflation"].items()]
        rows.append("\\midrule")
        rows.append("Leakage rate (\\%) & " + " & ".join(f1(100 * x) for x in d["leakage_rate"]))
        rows.append("New-lesion images & " + " & ".join(f"{x:.0f}" for x in d["new_lesion_images"]))
        out.append(table(f"Model {model}: per-client apparent inflation (BA points, five seeds; no standardization).",
                         f"tab:s3{model}", "lrrrrrr", "Method & 0 & 1 & 2 & 3 & 4 & 5", rows))
    rows = []
    for model, v in a["dose_response_loo"].items():
        for m, x in v.items():
            rows.append(f"{model} & {NAMES[m]} & {f1(100 * x['slope_all'])} & {f1(100 * x['slope_leave_one_out']['0'] if '0' in x['slope_leave_one_out'] else 100 * x['slope_leave_one_out'][0])} & "
                        f"{f1(100 * x['range_leave_one_out'][0])} to {f1(100 * x['range_leave_one_out'][1])}")
    out.append(table("Weighted least-squares slope of per-client inflation on leakage rate (points per 100\\% leakage): "
                     "all six clients, without client 0, and range when any one client is left out.", "tab:s3dr",
                     "llrrr", "Model & Method & All & Without 0 & Leave-one-out range", rows))
    return "".join(out)


def s5_s7():
    out = ["\\section*{S5. Federated conformal prediction: all rules, scores and levels}\n"]
    for title, fa, fb in (("in-federation lesion-grouped FedAvg models (E5)", "E5_conformal.json", "E5_conformal_pandermfrozen.json"),
                          ("LOCO models (E6, test 5b)", "E6_conformal_loco.json", "E6_conformal_loco_pandermfrozen.json")):
        rows = []
        for model, f in (("A", fa), ("B", fb)):
            d = load(f)
            for key in ("lac_a0.1", "aps_a0.1", "lac_a0.05", "lac_a0.2", "aps_a0.05", "aps_a0.2"):
                for m in ("global", "clientwise", "weighted", "fcp", "shrink", "eb"):
                    r = d[key][m]
                    rows.append(f"{model} & {key.replace('_a', ', $\\alpha$=').upper().replace('$\\ALPHA$', '$\\alpha$')} & {CONF[m]} & "
                                f"{f3(r['worst_center_q10'])} & {f3(min(r['mean_coverage']))} & {r['set_size_equal_weight']:.2f}")
                rows.append("\\midrule")
        out.append(table(f"Conformal results on {title}: worst-client q10, lowest client mean coverage and mean set size "
                         "(equal weight per client), mean over models.", f"tab:s5{fa[:2]}", "lllrrr",
                         "Model & Score, level & Rule & Worst q10 & Min mean cov. & Size", rows[:-1], size="\\scriptsize"))
        rows = []
        for model, f in (("A", fa), ("B", fb)):
            d = load(f)["lac_a0.1"]
            for m in ("global", "clientwise", "fcp", "eb"):
                rows.append(f"{model} & {CONF[m]} & " + " & ".join(f"{c:.3f}/{s:.2f}" for c, s in
                                                                    zip(d[m]["mean_coverage"], d[m]["mean_set_size"])))
        out.append(table(f"Per-client mean coverage / mean set size, {title} (LAC, $\\alpha=0.1$).", f"tab:s5pc{fa[:2]}",
                         "ll" + "r" * 6, "Model & Rule & 0 & 1 & 2 & 3 & 4 & 5", rows, size="\\scriptsize", wide=True))
    out.append("\\section*{S7. Conformal prediction on personalized models (descriptive)}\n")
    rows = []
    for run in ("fedbn_r9", "fedavgft_r9", "fedper_r9", "local_e20"):
        for model, suf in (("A", ""), ("B", "_pandermfrozen")):
            d = load(f"E7_conformal_{run}{suf}.json")
            if d is None:
                continue
            for m in ("global", "clientwise", "fcp", "eb"):
                r = d["lac_a0.1"][m]
                rows.append(f"{model} & {NAMES[run]} & {CONF[m]} & {f3(r['worst_center_q10'])} & {f3(min(r['mean_coverage']))} & "
                            f"{r['set_size_equal_weight']:.2f}")
    out.append(table("Personalized lesion-grouped models (LAC, $\\alpha=0.1$, five seeds).", "tab:s7", "lllrrr",
                     "Model & Method & Rule & Worst q10 & Min mean cov. & Size", rows, size="\\scriptsize"))
    return "".join(out)


def s6():
    d = load("Conformal_sensitivity.json")
    out = ["\\section*{S6. Sensitivity analyses of the conformal results (post hoc)}\n",
           "All on LAC, $\\alpha=0.1$. q10 from 1,000 draws; range of q10 over five independent sets of 100 draws; "
           "one image per lesion in both halves; global and FCP thresholds from exact order statistics; "
           "calibration-only q10 with five fixed evaluation halves per model and 100 lesion-bootstrap calibration sets; "
           "label-shift-weighted (LSW) comparator.\n"]
    if d is None:
        return out[0] + "Pending.\n"
    rows = []
    for key, r in d.items():
        model, exp = key.split("/")
        for m in ("global", "clientwise", "fcp", "eb", "lsw"):
            mc = [x[m]["worst_center_q10"] for x in r["mc_100x5"]]
            ex = r["exact_quantiles"].get(m, {}).get("worst_center_q10")
            co = r["calibration_only"].get(m, {}).get("worst_center_q10")
            rows.append(f"{model} & {exp.replace('_', ' ')} & {CONF[m]} & {f3(r['mc_1000'][m]['worst_center_q10'])} & "
                        f"{f3(min(mc))}--{f3(max(mc))} & {f3(r['one_per_lesion'][m]['worst_center_q10'])} & "
                        f"{f3(ex) if ex is not None else '--'} & {f3(co) if co is not None else '--'} & "
                        f"{f3(min(r['mc_1000'][m]['mean_coverage']))} & {f3(min(r['one_per_lesion'][m]['mean_coverage']))} & "
                        f"{r['mc_1000'][m]['set_size_equal_weight']:.2f}")
        rows.append("\\midrule")
    out.append(table("Worst-client q10 under alternative estimators, and lowest client mean coverage (all images; one image "
                     "per lesion).", "tab:s6", "lllrrrrrrrr",
                     "Model & Exp. & Rule & q10 (1,000) & 5$\\times$100 & 1 img./les. & Exact & Cal.-only & Min cov. & Min cov. 1 img. & Size",
                     rows[:-1], size="\\scriptsize", wide=True))
    rows = []
    for key, r in d.items():
        if "boot_identity_2000x20" not in r:
            continue
        model = key.split("/")[0]
        for name, lab in (("boot_identity_2000x20", "identity kept, 2,000 $\\times$ 20 draws"),
                          ("boot_identity_500x100", "identity kept, 500 $\\times$ 100 draws"),
                          ("boot_one_per_lesion_500x20", "one image per lesion, 500 $\\times$ 20 draws")):
            rows.append(f"{model} & {lab} & {p1(0, r[name]['clientwise']).split(' ', 1)[1]} & {p1(0, r[name]['global']).split(' ', 1)[1]}")
    out.append(table("95\\% intervals of the EB-ShrinkFCP gain in worst-client q10 over client-wise and global thresholds "
                     "on the LOCO models (points), from lesion bootstraps that keep resampled copies of a lesion together.",
                     "tab:s6boot", "llrr", "Model & Bootstrap & vs client-wise & vs global", rows))
    return "".join(out)


def nice(key):
    model, m = key.split("/")
    return f"{model} {NAMES.get(m, m)}"


def s8():
    d = load("Calibration_sensitivity.json")
    out = ["\\section*{S8. Sensitivity analyses of the calibration results (post hoc)}\n"]
    if d is None:
        return out[0] + "Pending.\n"
    lab = {"ece_unmatched": "ECE, unmatched (as in E3)", "ece_matched": "ECE, size- and composition-matched",
           "nll": "NLL (composition-reweighted)", "brier": "Brier (composition-reweighted)", "ece10": "ECE, 10 bins",
           "ece20": "ECE, 20 bins", "ece_mass15": "ECE, 15 equal-mass bins", "ece_mass15_matched": "ECE equal-mass, matched"}
    rows = []
    for key, r in d.items():
        for part, pl in (("main", "all new"), ("excluding_val_exposed", "excl. val.-exposed")):
            for k, (p, lo, hi) in r[part].items():
                sc = 1 if k == "nll" else 100
                rows.append(f"{nice(key)} & {pl} & {lab[k]} & "
                            f"{f'{sc * p:+.2f}'.replace('-', '$-$')} [{f'{sc * lo:+.2f}'.replace('-', '$-$')}, {f'{sc * hi:+.2f}'.replace('-', '$-$')}]")
        rows.append("\\midrule")
    out.append(table("New-lesion subset minus full image-level test set (ECE and Brier in points, NLL in nats); 95\\% "
                     "lesion-bootstrap intervals (500 replicates). Share of image-level test images exposed only through "
                     f"the validation fold: {100 * next(iter(d.values()))['val_only_share_of_test']:.1f}\\%.",
                     "tab:s8", "lllr", "Model, method & New subset & Metric & Difference [95\\% CI]", rows[:-1], size="\\scriptsize"))
    rows = []
    for key, r in d.items():
        for k, v in r["temperature_rule"].items():
            kind, rule = k.split("|")
            rows.append(f"{nice(key)} & {kind} & {'mean of client temperatures' if rule == 'mean' else 'pooled NLL-optimal'} & "
                        f"{v['T']:.3f} & {100 * v['ece']:.2f} & {v['nll']:.3f}")
    out.append(table("Global temperature: image-weighted mean of per-client temperatures (used) versus the NLL-optimal "
                     "temperature on the pooled validation set (mean of five seeds).", "tab:s8t", "lllrrr",
                     "Model, method & Split & Rule & T & ECE ($\\times$100) & NLL", rows, size="\\scriptsize"))
    return "".join(out)


def s10():
    a = load("Audit_supplement.json")
    out = ["\\section*{S10. Run matrix and seed-level results}\n"]
    rows = ["E0 & Official & A & FedAvg, FedProx, SCAFFOLD, pooled & 0--2",
            "E1 & Image, lesion & A, A-frozen, B & FedAvg, pooled (A: also FedAvg 27 rounds) & 0--2",
            "E2 & Image, lesion & A, B & FedAvg, FedProx, SCAFFOLD, FedBN (A), FedAvg-FT, pooled, local-only & 0--4",
            "E3 & Image, lesion & A, B & calibration of E2/E7 models; MC dropout (A FedAvg) & 0--4",
            "E5 & Lesion & A, B & conformal rules on FedAvg models & 0--4",
            "E6 & Lesion (LOCO) & A, B & FedAvg without one client (6 per seed); PAD-UFES-20 & 0--2",
            "E7 & Image, lesion & A, B & FedPer (A, B), conformal on personalized models & 0--4",
            "E8 & Lesion, 2 crossover arms & A, A-frozen, B & FedAvg, FedPer (A, B), pooled, local-only & 0--2",
            "E9 & Image, lesion & B & FedAvg, pooled, local-only with two-layer heads & 0--4"]
    out.append(table("Run matrix.", "tab:s10runs", "llllr", "Exp. & Splits & Models & Methods & Seeds", rows, wide=True))
    rows = []
    for key in sorted(a["seed_level_pooled_ba"]):
        kind, base = key.split("|")
        v = a["seed_level_pooled_ba"][key]
        model = "B" if base.startswith("pandermfrozen_") else "A"
        name = NAMES.get(base.replace("pandermfrozen_", ""), base)
        dd = a["paired_ba_minus_fedavg"].get(key)
        diff = "--" if dd is None else f"{100 * dd['mean']:+.1f} (SD {100 * (dd['sd'] or 0):.1f})".replace("-", "$-$")
        rows.append(f"{model} & {kind} & {name} & " + " & ".join(f"{v[str(s)] if str(s) in v else v.get(s, float('nan')):.3f}"
                                                               for s in range(5) if str(s) in v or s in v) + " " * 0
                    + "".join(" & " for _ in range(5 - len(v))) + f" & {diff}")
    out.append(table("Seed-level pooled balanced accuracy and paired difference to FedAvg of the same model and split type "
                     "(points; mean and SD over seeds).", "tab:s10seed", "lll" + "r" * 6,
                     "Model & Split & Method & s0 & s1 & s2 & s3 & s4 & $-$FedAvg", rows, size="\\scriptsize", wide=True))
    return "".join(out)


def s11():
    d = load("E8_sibling_exposure.json")
    out = ["\\section*{S11. Randomized sibling exposure (E8)}\n"]
    if d is None:
        return out[0] + "Pending: results of E8 are not yet available.\n"
    rows = []
    for run, r in d["families"].items():
        e, ci = r["effect"], r["ci"]
        rows.append(f"{r['encoder']} & {NAMES.get(run, run)} & {p1(e['ba'], ci['ba'])} & {p1(e['acc'], ci['acc'])} & "
                    f"{p1(e['mel_recall'], ci['mel_recall'])} & {f3(e['nll'])} & {p1(e['brier'], ci['brier'])} & "
                    f"{p1(r['sanity_nontarget_ba_arm1_minus_arm2'])}")
    out.append(table("Exposure effect on fixed target images (exposed minus unexposed; points except NLL), seeds 0--2, "
                     "95\\% paired lesion-bootstrap intervals; last column: sanity check on non-target test images "
                     "(arm 1 minus arm 2).", "tab:s11", "llrrrrrr",
                     "Model & Method & BA & Accuracy & MEL recall & NLL & Brier & Sanity BA", rows, size="\\scriptsize", wide=True))
    rows = [f"{k} & {p1(v['effect'], v['ci'])}" for k, v in d["contrasts"].items()]
    out.append(table("Difference in BA exposure effect relative to FedAvg of the same encoder (points).", "tab:s11c", "lr",
                     "Contrast & Difference [95\\% CI]", [r.replace("_", "\\_") for r in rows], size="\\scriptsize"))
    return "".join(out)


def s12():
    d = load("E9_analysis_panderm.json")
    out = ["\\section*{S12. Capacity-matched two-layer heads for model B (E9)}\n"]
    if d is None:
        return out[0] + "Pending: results of E9 are not yet available.\n"
    rows = []
    for j, m in enumerate(d["methods"]):
        rows.append(f"{NAMES.get(m, m)} & {d['point']['pooled']['mean_ba']['image'][j]:.3f} & "
                    f"{d['point']['pooled']['mean_ba']['lesion'][j]:.3f} & "
                    f"{f1(100 * (d['point']['pooled']['mean_ba']['image'][j] - d['point']['pooled']['mean_ba']['lesion'][j]))}")
    out.append(table("Model B pooled BA by method on image-level and lesion-grouped splits (five seeds), including two-layer "
                     "heads for FedAvg, pooled and local-only training; last column: retrained-split difference (points).",
                     "tab:s12", "lrrr", "Method & Image-level & Lesion-grouped & Difference", rows))
    tb = d["point"]["pooled"]["tau_between"]
    out.append(f"Kendall's $\\tau_b$ between the two rankings with the two-layer heads included: {tb:.2f}.\n")
    return "".join(out)


def s9():
    rows = [("v0.1--v0.3", "26--27 Oct", "none", "Research questions, splits, models, metrics"),
            ("v0.4--v0.6", "27--28 Oct", "none", "Gate 1 and 2, primary outcome, H3b, $\\tau$ noise floor, A-frozen control (reviews 1--2)"),
            ("v0.7", "28 Oct", "none", "RQ4 and ShrinkFCP added"),
            ("v0.8--v0.10", "29--30 Oct", "none", "H3b clients 0, 2, 3; worst-client q10 metric; guards; per-encoder success (reviews 3--5)"),
            ("e0 \\ldots{} e5 tags", "30 Oct--3 Nov", "E0, E0b, E1, E2, E5", "pre-run tags before each batch"),
            ("v0.11", "3 Nov", "E0--E5 (RQ4 failed)", "EB-ShrinkFCP and test 5b on LOCO models (before any LOCO model)"),
            ("e7, PAD, NN tags", "5--6 Nov", "E0--E6", "pre-run tags; PAD interpretation rule; NN analysis exploratory"),
            ("v0.12", "7 Nov", "E0--E7, PAD, NN; external review", "E8, E9 and their tests; post hoc sensitivity analyses")]
    out = ["\\section*{S9. Protocol chronology and deviations}\n",
           table("Protocol versions and pre-run tags (git, pushed to GitHub), with the results available when each was "
                 "written.", "tab:s9", "llp{3.6cm}p{6.2cm}", "Version / tag & Date (2026) & Results available & Content",
                 [f"{a} & {b} & {c} & {d}" for a, b, c, d in rows], size="\\scriptsize", wide=True)]
    dev = ["FLamby's published FedAvg value recorded after E0 (not used in Gate 1)",
           "PyTorch 2.5.1 instead of 2.4.1",
           "Frozen-encoder embeddings centered with the federated training mean (31 Oct, after the first convergence check, "
           "before any test result)",
           "No five-member deep ensemble (seeds use different splits)",
           "Personalized FCP not implemented (threshold rule not verifiable, no code)",
           "Optional E4 and the original E8 (model C) not run; identifiers E8/E9 reused in v0.12",
           "U not re-tuned for the two-layer FedPer-style head (fixed before E7)",
           "Post hoc sensitivity analyses added after the external review (S6, S8; listed in v0.12)"]
    out.append("Deviations: " + "; ".join(f"({i + 1}) {x}" for i, x in enumerate(dev)) + ".\n")
    return "".join(out)


def s2():
    return r"""\section*{S2. Implementation details}
\begin{itemize}
\item \textbf{Model A:} EfficientNet-B0 (efficientnet-pytorch, ImageNet weights), new 8-class linear output layer;
FedAvg, FedProx, SCAFFOLD, FedBN and FedPer: 9 rounds $\times$ 100 local SGD steps (batch 64, learning rate 0.01, mixed
precision); pooled and local-only: Adam, learning rate $5\times10^{-4}$, batch 64, 20 epochs; weighted focal loss
($\gamma=2$) with FLamby's fixed class weights (5.58, 2.05, 7.02, 26.12, 9.54, 101.07, 92.52, 38.34 for MEL, NV, BCC, AK,
BKL, DF, VASC, SCC).
\item \textbf{Aggregation:} all trainable parameters averaged with weights proportional to training images.
Batch-normalization running statistics are buffers and stay with each client in every method; FedBN keeps the
batch-normalization weights and biases local as well; FedPer (model A) keeps the output layer local.
\item \textbf{Evaluation model:} global methods are evaluated with the copy of the first participating client,
including its running statistics (FLamby convention); personalized methods with each client's own model; LOCO
models apply the running statistics of the first participating client, with no adaptation to the held-out client.
\item \textbf{Frozen encoders:} embeddings in evaluation mode, fp16; one plain view and five augmented views per image
(seed $1{,}000{,}003\,v+i$); centering with the federated training mean (training fold only, all five views,
held-out client excluded for LOCO); heads with PyTorch default initialization (Kaiming-uniform weights); FedAvg-type
heads: 9 rounds $\times$ $U=6{,}400$ SGD steps; FedAvg-FT: three local SGD steps from the final FedAvg head.
\item \textbf{Temperature scaling:} per-client NLL-optimal temperature on validation lesions (grid of 600 values from
0.05 to 20), global temperature = their mean weighted by validation images.
\item \textbf{Conformal histograms:} 1,000 equal-width bins on $[0,1]$; the threshold is the upper edge of the bin in
which the cumulative count reaches the required rank (conservative).
\item \textbf{Bootstraps:} test lesions resampled with replacement within each seed's test set; all models and methods
of a seed share the replicate; seeds averaged inside each replicate.
\end{itemize}
"""


def main():
    doc = r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs,graphicx,amsmath,amssymb}
\usepackage[T1]{fontenc}
\usepackage{url}
\usepackage{mathptmx}
\graphicspath{{../figures/}{figures/}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\title{Supplementary material\\[4pt]\large Lesion-level leakage and site-level reliability in Fed-ISIC2019}
\date{}
\begin{document}
\maketitle
All tables are generated by \texttt{scripts/make\_supplement.py} from the result files in the code repository
(\url{https://github.com/Sisty37/reliable-fed-derm}).
""" + s1() + s2() + s3() + s5_s7() + s6() + s8() + s9() + s10() + s11() + s12() + r"""
\section*{Figure S1}
\begin{figure}[!htbp]
\centering
\IfFileExists{../figures/FigS1_proximity_examples.pdf}{\includegraphics[width=\textwidth]{FigS1_proximity_examples}}{\fbox{FigS1\_proximity\_examples.pdf}}
\caption{Exploratory nearest-neighbor examples (randomly drawn leaked test images, seed 20261106, not selected): each
test image with its nearest training image in the frozen embedding space.}
\end{figure}
\end{document}
"""
    os.makedirs("paper", exist_ok=True)
    open(os.path.join("paper", "supplement.tex"), "w", encoding="utf-8").write(doc)
    print("saved paper/supplement.tex")


if __name__ == "__main__":
    main()
