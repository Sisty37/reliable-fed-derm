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
    out = ["\\section*{S4. Federated conformal prediction: all rules, scores and levels}\n"]
    for title, fa, fb in (("in-federation lesion-grouped FedAvg models (E5)", "E5_conformal.json", "E5_conformal_pandermfrozen.json"),
                          ("LOCO models (E6, test 5b)", "E6_conformal_loco.json", "E6_conformal_loco_pandermfrozen.json")):
        rows = []
        for model, f in (("A", fa), ("B", fb)):
            d = load(f)
            for key in ("lac_a0.1", "aps_a0.1", "lac_a0.05", "lac_a0.2", "aps_a0.05", "aps_a0.2"):
                for m in ("global", "clientwise", "weighted", "fcp", "shrink", "eb"):
                    r = d[key][m]
                    klab = key.replace('_a', ', $\\alpha$=').upper().replace('$\\ALPHA$', '$\\alpha$')
                    rows.append(f"{model} & {klab} & {CONF[m]} & "
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
    out.append("\\section*{S6. Conformal prediction on personalized models (descriptive)}\n")
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
    out = ["\\section*{S5. Sensitivity analyses of the conformal results (post hoc)}\n",
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
    out = ["\\section*{S7. Sensitivity analyses of the calibration results (post hoc)}\n"]
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
    out = ["\\section*{S9. Run matrix and seed-level results}\n"]
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
    out = ["\\section*{S10. Randomized sibling exposure (E8)}\n"]
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
    out = ["\\section*{S11. Capacity-matched two-layer heads for model B (E9)}\n"]
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



def s13():
    """Fed-ISIC2019 studies and their evaluation split (checked online 7 Nov 2026; sources listed below the table)."""
    rows = [
        ("du Terrail et al. 2022 (NeurIPS D\\&B) [a]", "defines the benchmark; image-level re-split of the 23,247 images", "BA; strategies shown in a figure, FL between local and pooled"),
        ("Tavakoli et al. 2024 (MLHC, PMLR 252) [b]", "FLamby split (per-client sizes reproduced from FLamby)", "FedAvg 0.657, central 0.683, Ditto 0.703, FENDA-FL 0.607"),
        ("Li et al. 2024 (ICLR, FedCompass) [c]", "FLamby default settings (split not named)", "time to 50\\% BA; no final BA comparison"),
        ("Allouah et al. 2024 (NeurIPS) [d]", "FLamby test set halved into validation/test", "FedAvg 0.731, FedProx 0.750, one-shot FENS 0.579"),
        ("Siomos et al. 2024 (arXiv) [e]", "``improved version'' of the FLamby benchmark", "FedAvg 59.5--74.8, FedPer up to 84.9 (architectures/settings)"),
        ("Ayromlou \\& Emerson 2025 (arXiv) [f]", "FLamby settings, per-client held-out test sets", "FedAvg 64.1, Ditto 71.4, Ditto + constraint up to 72.2"),
        ("Tupper \\& Gagn\\'e 2025 (arXiv) [g]", "same test set as FLamby; 80/20 stratified train/validation", "FedIT 0.656, best (FedSA) 0.737"),
    ]
    out = ["\\section*{S12. Fed-ISIC2019 studies and their evaluation split}\n",
           "Studies found in a targeted search (not a systematic review). None reported a lesion-grouped split; all "
           "evaluated on the FLamby test set or stated that they followed FLamby's settings. Differences between "
           "the compared methods in these studies are often a few balanced-accuracy points, comparable to the "
           "apparent leakage effect measured here.\n",
           table("Fed-ISIC2019 evaluations (BA, balanced accuracy, as reported by each study).", "tab:s13",
                 "p{4.2cm}p{5.0cm}p{6.3cm}", "Study & Split as described & Reported results",
                 [f"{a} & {b} & {c}" for a, b, c in rows], size="\\scriptsize", wide=True),
           "\\noindent{\\footnotesize [a] arXiv:2210.04620. [b] arXiv:2309.16825; PMLR 252. [c] arXiv:2309.14675; ICLR 2024. "
           "[d] NeurIPS 2024, ``Revisiting ensembling in one-shot federated learning''. [e] arXiv:2410.02006. "
           "[f] arXiv:2505.07525. [g] arXiv:2510.12741.}\n"]
    return "".join(out)


CC_RULES = {"global": "Global", "clientwise": "Client-wise", "eb": "EB-ShrinkFCP",
            "mondrian_pooled": "Mondrian (pooled)", "mondrian_client": "Mondrian (per client)"}
CLS8 = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


def s14():
    d = load("Class_conditional.json")
    out = ["\\section*{S13. Coverage by diagnostic class (post hoc)}\n",
           "LAC, $\\alpha=0.1$, FedAvg models, 200 lesion-wise 50/50 draws per model; coverage by true class on the "
           "evaluation halves, pooled over clients. Mondrian rules use one threshold per class (from the pooled "
           "calibration scores of that class, or per client with a fallback to the pooled class threshold when a "
           "client has fewer than 10 calibration images of the class).\n"]
    if d is None:
        return out[0] + "Pending.\n"
    rows = []
    for key, v in d["isic"].items():
        model, exp = key.split("/")
        for r, lab in CC_RULES.items():
            x = v[r]
            rows.append(f"{model} & {'LOCO' if 'loco' in exp else 'E5'} & {lab} & " + " & ".join(f3(c) for c in x["cov_by_class"])
                        + f" & {f3(x['coverage'])} & {x['size']:.2f}")
        rows.append("\\midrule")
    out.append(table("Fed-ISIC2019: coverage by class, overall coverage and mean set size.", "tab:s14", "lll" + "r" * 10,
                     "Model & Exp. & Rule & " + " & ".join(CLS8) + " & All & Size", rows[:-1], size="\\scriptsize", wide=True))
    rows = []
    for model, v in d["pad"].items():
        for r in ("marginal", "mondrian"):
            rows.append(f"{model} & {'Marginal' if r == 'marginal' else 'Mondrian'} & "
                        + " & ".join(f3(c) for c in v[r]["cov_by_class"]) + f" & {f3(v[r]['coverage'])} & {v[r]['size']:.2f}")
    out.append(table("PAD-UFES-20, recalibrated on half of the patients (100 draws per seed, seeds 0--4): coverage by class.",
                     "tab:s14pad", "ll" + "r" * 8, "Model & Rule & " + " & ".join(d["pad"]["A"]["classes"]) + " & All & Size",
                     rows, size="\\scriptsize"))
    return "".join(out)


EBV = {"clientwise": "Client-wise", "global": "Global", "prespecified": "EB-ShrinkFCP (pre-specified)",
       "re_mean": "EB, random-effects mean", "floor_1e-8": "EB, variance floor $10^{-8}$",
       "floor_1e-4": "EB, variance floor $10^{-4}$", "no_design": "EB, no images-per-lesion factor",
       "boot_var": "EB, lesion-bootstrap variance"}


def s15():
    d = load("EB_variants.json")
    out = ["\\section*{S14. Variants of EB-ShrinkFCP (post hoc)}\n",
           "LAC, $\\alpha=0.1$, FedAvg models, 1,000 lesion-wise draws per model. Variants change one element of the "
           "pre-specified rule at a time.\n"]
    if d is None:
        return out[0] + "Pending.\n"
    rows = []
    for key, v in d.items():
        model, exp = key.split("/")
        for r, lab in EBV.items():
            if r in v:
                x = v[r]
                rows.append(f"{model} & {'LOCO' if 'loco' in exp else 'E5'} & {lab} & {f3(x['worst_center_q10'])} & "
                            f"{f3(x['min_mean_coverage'])} & {x['set_size']:.2f}")
        rows.append("\\midrule")
    out.append(table("Worst-client q10, lowest client mean coverage and mean set size.", "tab:s15", "lllrrr",
                     "Model & Exp. & Rule & Worst q10 & Min mean cov. & Size", rows[:-1], size="\\scriptsize"))
    rows = [f"{k.split('/')[0]} & {'LOCO' if 'loco' in k else 'E5'} & "
            + ", ".join(f"{x:.2f}" for x in v["_boot_over_spacing_variance_median_by_client"])
            for k, v in d.items() if "_boot_over_spacing_variance_median_by_client" in v]
    out.append(table("Lesion-bootstrap variance of each client's threshold divided by the variance used by EB-ShrinkFCP "
                     "(median over draws; clients 0--5). Values below 1 mean that the pre-specified approximation is "
                     "conservative, which increases shrinkage.", "tab:s15v", "llp{7cm}", "Model & Exp. & Ratio by client", rows,
                     size="\\scriptsize"))
    return "".join(out)


SIMR = {"clientwise": "Client-wise", "global": "Global", "eb": "EB-ShrinkFCP", "eb_re_mean": "EB, random-effects mean"}


def s16():
    d = load("Conformal_simulation.json")
    out = ["\\section*{S15. Simulation with clustered scores and calibration-budget curves (post hoc)}\n"]
    if d is None:
        return out[0] + "Pending.\n"
    out.append("Six clients with " + ", ".join(str(n) for n in d["n_lesions"]) + " calibration lesions; each lesion has "
               "$1+\\mathrm{Poisson}(1)$ images; latent score $z=\\delta_c+u_\\ell+e_i$ with lesion effect "
               "$u_\\ell\\sim N(0,\\rho)$, image noise $e_i\\sim N(0,1-\\rho)$ and client shift $\\delta_c\\sim N(0,\\tau^2)$ "
               "drawn per replicate; conformal score $1/(1+e^{-z})$. The true coverage of each threshold for one image of a "
               "new lesion is computed from 200,000 draws of the known client distribution; 2,000 replicates per "
               "scenario.\n")
    rows = []
    for sc, v in d["scenarios"].items():
        tau, rho = sc.replace("tau", "").split("_rho")
        for r, lab in SIMR.items():
            x = v[r]
            rows.append(f"{tau} & {rho} & {lab} & {f3(x['worst_q10'])} & {f3(x['worst_mean'])} & {x['worst_p_below_085']:.2f}")
        rows.append("\\midrule")
    out.append(table("True coverage in the simulation: worst client's 10th percentile over calibration samples, lowest "
                     "client mean, and highest client probability of true coverage below 0.85.", "tab:s16", "lllrrr",
                     "$\\tau$ & $\\rho$ & Rule & Worst q10 & Min mean & Max P($<$0.85)", rows[:-1], size="\\scriptsize"))
    rows = [f"{k.replace('rho', '')} & " + " & ".join(f"{x:.2f}" for x in v["ratio_approx_over_true"])
            for k, v in d["variance_check"].items()]
    out.append(table("Variance used by EB-ShrinkFCP divided by the actual sampling variance of the client quantile "
                     "(fixed client distributions, 2,000 replicates), clients ordered from largest to smallest.",
                     "tab:s16v", "l" + "r" * len(d["n_lesions"]),
                     "$\\rho$ & " + " & ".join(f"{n} les." for n in d["n_lesions"]), rows, size="\\scriptsize"))
    rows = []
    for model, v in d["budget_curves"].items():
        for b, w in v.items():
            rows.append(f"{model} & {b} & " + " & ".join(f"{f3(w[r]['worst_q10'])} / {f3(w[r]['min_mean_cov'])}"
                                                      for r in ("clientwise", "global", "eb")))
        rows.append("\\midrule")
    out.append(table("Calibration-budget curves on the LOCO FedAvg models: at most the given number of calibration lesions "
                     "per client (all available if fewer), evaluation on the other half; worst-client q10 / lowest client "
                     "mean coverage, 300 draws per model.", "tab:s16b", "llrrr",
                     "Model & Lesions per client & Client-wise & Global & EB-ShrinkFCP", rows[:-1], size="\\scriptsize"))
    return "".join(out)


BUDGET_NAMES = {"fedavg_r9": "FedAvg, 9 rounds", "fedavg_r27": "FedAvg, 27 rounds", "fedavgft1_r9": "FedAvg-FT, 1 step",
                "fedavgft_r9": "FedAvg-FT, 3 steps", "fedavgft10_r9": "FedAvg-FT, 10 steps", "fedavgft30_r9": "FedAvg-FT, 30 steps",
                "fedavgft100_r9": "FedAvg-FT, 100 steps", "pooled_e5": "Pooled, 5 epochs", "pooled_e20": "Pooled, 20 epochs",
                "pooled_e60": "Pooled, 60 epochs", "local_e5": "Local-only, 5 epochs", "local_e20": "Local-only, 20 epochs",
                "local_e60": "Local-only, 60 epochs"}


def s17nn():
    d = load("NN_matched.json")
    out = ["\\section*{S16. Nearest-neighbor proximity matched by client and class (post hoc)}\n",
           "Frozen plain-view embeddings, centered with the training mean of each split and L2-normalized; image-level "
           "splits, seeds 0--4. Matched differences average the within-stratum (client $\\times$ class) differences, "
           "weighted by the stratum's new-lesion count (strata with at least three leaked and three new images). "
           "1-NN inflation is standardized like Eq.~(1); ``siblings removed'' excludes all training images of the "
           "test image's lesion from the neighbor search.\n"]
    if d is None:
        return ""
    lab = {"effb0": "A-frozen", "panderm": "B"}
    rows = []
    for enc, v in d.items():
        m, sd = v["mean"], v["sd"]
        rows.append(f"{lab.get(enc, enc)} & {m['sim_raw_diff']:+.3f} & {m['sim_matched_diff']:+.3f} ({sd['sim_matched_diff']:.3f}) & "
                    f"{m['sim_other_lesion_raw_diff']:+.3f} & {m['sim_other_lesion_matched_diff']:+.3f} ({sd['sim_other_lesion_matched_diff']:.3f}) & "
                    f"{100 * m['nn1_std_inflation']:+.1f} ({100 * sd['nn1_std_inflation']:.1f}) & "
                    f"{100 * m['nn1_no_sibling_std_inflation']:+.1f} ({100 * sd['nn1_no_sibling_std_inflation']:.1f})")
    out.append(table("Leaked minus new-lesion test images: cosine similarity to the nearest training image and to the nearest "
                     "training image of another lesion (raw and client $\\times$ class matched), and standardized 1-NN "
                     "inflation in points with and without siblings in the training set; mean (SD) over seeds.", "tab:s17nn",
                     "lrrrrrr", "Model & NN raw & NN matched & Other-lesion raw & Other-lesion matched & 1-NN infl. & 1-NN, no siblings",
                     rows, size="\\scriptsize", wide=True))
    return "".join(out)


def s17():
    d = load("Budget_controls.json")
    out = ["\\section*{S17. Training-budget controls (post hoc)}\n"]
    if d is None or not any(len(v) > 4 for v in d.values()):
        return ""                                        # optional runs not performed: section omitted
    rows = []
    for model, v in d.items():
        for m, r in v.items():
            rows.append(f"{model} & {BUDGET_NAMES.get(m, m)}{'$^*$' if r['protocol_setting'] else ''} & "
                        f"{p1(r['inflation'], r['ci'])} & {p1(r['minus_fedavg_r9'], r['minus_fedavg_r9_ci'])} & "
                        f"{f3(r['ba_image'])} & {f3(r['ba_lesion'])}")
        rows.append("\\midrule")
    out.append(table("Apparent inflation in clients 0, 2 and 3 (points, 95\\% lesion-bootstrap intervals), its difference to "
                     "FedAvg with 9 rounds, and pooled BA on image-level and lesion-grouped splits, seeds 0--2. $^*$protocol setting.",
                     "tab:s17", "llrrrr", "Model & Training & Inflation & $-$FedAvg & BA image & BA lesion", rows[:-1],
                     size="\\scriptsize"))
    return "".join(out)


def s9():
    rows = [("v0.1--v0.3", "26--27 Oct", "none", "Research questions, splits, models, metrics"),
            ("v0.4--v0.6", "27--28 Oct", "none", "Gate 1 and 2, primary outcome, H3b, $\\tau$ noise floor, A-frozen control (reviews 1--2)"),
            ("v0.7", "28 Oct", "none", "RQ4 and ShrinkFCP added"),
            ("v0.8--v0.10", "29--30 Oct", "none", "H3b clients 0, 2, 3; worst-client q10 metric; guards; per-encoder success (reviews 3--5)"),
            ("e0 \\ldots{} e5 tags", "30 Oct--3 Nov", "E0, E0b, E1, E2, E5", "pre-run tags before each batch"),
            ("v0.11", "3 Nov", "E0--E5 (RQ4 failed)", "EB-ShrinkFCP and test 5b on LOCO models (before any LOCO model)"),
            ("e7, PAD, NN tags", "5--6 Nov", "E0--E6", "pre-run tags; PAD interpretation rule; NN analysis exploratory"),
            ("v0.12", "7 Nov", "E0--E7, PAD, NN; external review", "E8, E9 and their tests; post hoc sensitivity analyses"),
            ("8 Nov (no new version)", "8 Nov", "all results, incl. E8, E9", "post hoc only: S12--S17")]
    out = ["\\section*{S8. Protocol chronology and deviations}\n",
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
""" + s1() + s2() + s3() + s5_s7().split("\\section*{S6.")[0] + s6() + "\\section*{S6." + s5_s7().split("\\section*{S6.")[1] + s8() + s9() + s10() + s11() + s12() + s13() + s14() + s15() + s16() + s17nn() + s17() + r"""
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
