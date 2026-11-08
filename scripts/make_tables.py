"""
Main-paper tables (Table I-IV) as LaTeX (booktabs, IEEE two-column layout), built only from the
result files in results/, splits/ and runs/split_official/ (no hand-typed numbers).

Usage (from the repo folder):  python scripts\\make_tables.py
Writes tables/Table1_dataset.tex ... tables/Table4_conformal.tex and tables/tables_preview.tex
(a stand-alone document that \\input's all four, for checking with pdflatex).
LaTeX packages needed in the paper: booktabs, multirow.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

R, OUT = "results", "tables"
NAMES = {"fedavg_r9": "FedAvg", "fedprox_r9": "FedProx", "scaffold_r9": "SCAFFOLD", "fedbn_r9": "FedBN",
         "fedavgft_r9": "FedAvg-FT", "fedper_r9": "FedPer", "pooled_e20": "Pooled", "local_e20": "Local-only"}
ORDER = ["fedavg_r9", "fedprox_r9", "scaffold_r9", "fedbn_r9", "fedavgft_r9", "fedper_r9", "pooled_e20", "local_e20"]


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def pts(x, ci=None, sign=True):
    """Value in points (x100) with one decimal and an optional 95% interval."""
    f = "{:+.1f}" if sign else "{:.1f}"
    s = f.format(100 * x)
    if ci is not None:
        s += " [" + f.format(100 * ci[0]) + ", " + f.format(100 * ci[1]) + "]"
    return s.replace("-", "$-$")


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(body)
    print(f"saved {OUT}/{name}")


# ---------------------------------------------------------------- Table I: clients
SOURCE = {0: "BCN20000", 1: "HAM vidir\\_molemax", 2: "HAM vidir\\_modern", 3: "HAM rosendahl", 4: "MSK",
          5: "HAM vienna\\_dias"}


def table1():
    sp = pd.read_csv(os.path.join("splits", "split_official.csv"))
    au = load("Audit_supplement.json")
    off = {r["client"]: r for r in au["leakage_official"]}
    pc = load("E7_analysis.json")["point"]["per_center"]
    rows = []
    for c in range(6):
        g = sp[sp.center == c]
        lo = off[c]["leaked_same_client"] + off[c]["leaked_other_client_only"]
        rows.append(f"{c} & {SOURCE[c]} & {len(g):,} & {g.lesion_id.nunique():,} & {len(g) / g.lesion_id.nunique():.2f} & "
                    f"{100 * lo:.1f} & {100 * pc['leakage_rate'][c]:.1f} & {pc['new_lesion_images'][c]:.0f} \\\\")
    lo = off["all"]["leaked_same_client"] + off["all"]["leaked_other_client_only"]
    tot = (f"All & & {len(sp):,} & {sp.lesion_id.nunique():,}$^a$ & {len(sp) / sp.lesion_id.nunique():.2f} & "
           f"{100 * lo:.1f} & {100 * np.average(pc['leakage_rate'], weights=[off[c]['test_images'] for c in range(6)]):.1f} & "
           f"{sum(pc['new_lesion_images']):,.0f} \\\\")
    body = r"""\begin{table}[!t]
\centering
\caption{Fed-ISIC2019 clients (``centers'' in FLamby), defined by acquisition source; clients 1, 2 and 5
come from one institution with different devices. Leakage: share of a client's image-level test images
whose lesion also has images in training, in the official FLamby split and as the mean over five
image-level re-splits; new test: mean number of new-lesion test images in the re-splits.}
\label{tab:dataset}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llrrrrrr@{}}
\toprule
 & & & & & \multicolumn{2}{c}{Leakage (\%)} & \\
\cmidrule(lr){6-7}
Client & Source & Images & Lesions & Img./lesion & Official & Re-splits & New test \\
\midrule
""" + "\n".join(rows) + "\n\\midrule\n" + tot + r"""
\bottomrule
\multicolumn{8}{@{}p{0.95\linewidth}@{}}{\footnotesize $^a$Nine lesions have images in clients 1 and 2, so the
per-client lesion counts sum to 11,856.}
\end{tabular}
\end{table}
"""
    write("Table1_dataset.tex", body)


def cs_text(model):
    """Corrected EB-ShrinkFCP gain over client-wise on the LOCO models: 1,000-draw point estimate and
    identity-preserving bootstrap interval (500 x 100 draws), from results/Conformal_sensitivity.json."""
    d = load("Conformal_sensitivity.json")[f"{model}/E6_loco"]
    g = d["mc_1000"]["eb"]["worst_center_q10"] - d["mc_1000"]["clientwise"]["worst_center_q10"]
    lo, hi = d["boot_identity_500x100"]["clientwise"]
    return f"{100 * g:+.1f} [{100 * lo:.1f}, {100 * hi:.1f}]".replace("-", "$-$")


# ---------------------------------------------------------------- Table V: randomized sibling exposure (E8)
def table5():
    e8, fu = load("E8_sibling_exposure.json"), load("Followup_analyses.json")["e8_seeds"]
    nm = [("fedavg_r9", "A (fine-tuned)", "FedAvg"), ("effb0frozen_fedavg_r9", "A-frozen", "FedAvg"),
          ("pandermfrozen_fedavg_r9", "B (PanDerm)", "FedAvg"), ("local_e20", "A (fine-tuned)", "Local-only"),
          ("pandermfrozen_fedper_r9", "B (PanDerm)", "FedPer-style head")]
    rows = []
    for run, enc, meth in nm:
        f = e8["families"][run]
        seeds = ", ".join(f"{100 * fu[run][s]['ba_effect']:+.1f}" for s in ("0", "1", "2")).replace("-", "$-$")
        san = ", ".join(f"{100 * fu[run][s]['sanity_nontarget']:+.1f}" for s in ("0", "1", "2")).replace("-", "$-$")
        rows.append(f"{enc} & {meth} & {pts(f['effect']['ba'], f['ci']['ba'])} & {seeds} & {san} \\\\")
    body = r"""\begin{table}[!t]
\centering
\caption{Randomized sibling exposure (E8): balanced accuracy of the fixed target images with their siblings in
training minus without (points, seeds 0--2), with 95\% lesion-bootstrap intervals that condition on the fitted
models and realized allocations; seed-level effects; and the non-target sanity difference between arms.
All methods in Supplementary~S10.}
\label{tab:exposure}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}llccc@{}}
\toprule
Encoder & Method & Effect [95\% CI] & Seeds 0, 1, 2 & Non-target \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("Table5_exposure.tex", body)


def pad_transfer(pad, key):
    """Transferred LAC coverage at alpha 0.1 on PAD-UFES-20 with its patient-bootstrap interval."""
    t = pad.get(key, {}).get("lac_a0.1", {}).get("transfer")
    return None if t is None else (t["coverage"], t["coverage_ci95"])


# ---------------------------------------------------------------- Table II: scorecard of pre-specified tests
def table2():
    e0 = [json.load(open(f))["bacc_pooled"] for f in sorted(glob.glob(os.path.join("runs", "split_official", "fedavg_r9_s*", "metrics.json")))]
    g1 = float(np.mean(e0))
    gA = load("E1_gate2.json")["fedavg_r9 (PRIMARY)"]
    hA, hB = load("E7_analysis.json"), load("E7_analysis_panderm.json")
    e5A, e5B = load("E5_conformal.json")["tests"], load("E5_conformal_pandermfrozen.json")["tests"]
    e6A, e6B = load("E6_conformal_loco.json")["tests"], load("E6_conformal_loco_pandermfrozen.json")["tests"]

    def verdict(ok):
        return r"\textbf{met}" if ok else "not met"

    def h3(d):
        ci = d["ci95"]["h3b.difference"]
        return pts(d["point"]["h3b"]["difference"], ci), ci[0] > 0

    hAv, hAok = h3(hA)
    hBv, hBok = h3(hB)
    rows = [
        ("Gate 1", "Pipeline: FedAvg BA, official split (3 seeds)", "A", f"{g1:.3f} vs 0.657 $\\pm$ 0.05",
         verdict(abs(g1 - 0.657) <= 0.05)),
        ("Gate 2 (RQ2)", "Apparent inflation $\\ge 2$, CI $>0$, FedAvg (primary)", "A", pts(gA["inflation"], gA["ci95"]),
         verdict(gA["inflation"] >= 0.02 and gA["ci95"][0] > 0)),
        ("H3b (RQ3)", "Personalized minus FedAvg inflation, clients 0, 2, 3", "A", hAv, verdict(hAok)),
        ("", "", "B", hBv, verdict(hBok)),
        ("RQ4", "ShrinkFCP, in-federation models: q10 gain $\\ge 2$ + guards", "A",
         pts(e5A["shrink"]["gain"], e5A["shrink"]["ci95"]) + "$^\\dagger$", verdict(e5A["shrink"]["success"])),
        ("", "", "B", pts(e5B["shrink"]["gain"], e5B["shrink"]["ci95"]) + "$^\\dagger$", verdict(e5B["shrink"]["success"])),
        ("5b (RQ4)", "EB-ShrinkFCP on LOCO models, same rule", "A", pts(e6A["eb"]["gain"], e6A["eb"]["ci95"]) + "$^\\dagger$",
         ("met" if e6A["eb"]["success"] else "not met") + "$^\\dagger$"),
        ("", "", "B", pts(e6B["eb"]["gain"], e6B["eb"]["ci95"]) + "$^\\dagger$",
         ("met" if e6B["eb"]["success"] else "not met") + "$^\\dagger$"),
    ]
    pad = load("E6_padufes.json") if os.path.exists(os.path.join(R, "E6_padufes.json")) else None
    if pad is not None:
        for mdl, key in (("A", "A"), ("B", "B")):
            t = pad_transfer(pad, key)
            if t is not None:
                cov, ci = t
                rows.append(("PAD (RQ1)" if mdl == "A" else "", "Transferred coverage fails ($<0.90$, CI excl.\\ 0.90)" if mdl == "A" else "",
                             mdl, f"{cov:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]", r"\textbf{met}" if ci[1] < 0.90 else "not met"))
    e8 = load("E8_sibling_exposure.json") if os.path.exists(os.path.join(R, "E8_sibling_exposure.json")) else None
    if e8 is not None and "primary" in e8:
        p8 = e8["primary"]
        rows.append(("E8 (RQ2)", "Exposure effect $\\ge 2$, CI $>0$, FedAvg", "A", pts(p8["effect"], p8["ci"]), verdict(p8["met"])))
    lines = [f"{a} & {b} & {m} & {v} & {r} \\\\" for a, b, m, v, r in rows]
    body = r"""\begin{table*}[!t]
\centering
\caption{All pre-specified tests, in the order they were registered. Effects in balanced-accuracy points
(Gates, H3b, E8) or coverage points (RQ4, 5b), with 95\% lesion-bootstrap intervals (patients for
PAD-UFES-20). RQ4 and 5b require the gain for \emph{each} of models A and B; the reference is client-wise
calibration. Gate 1 compares with the FedAvg baseline reported in the FENDA-FL study
\cite{tavakoli2024fendafl}; meeting it is a replication tolerance. Gate 2 is the single primary outcome;
the other tests are secondary and not adjusted for multiplicity. $^\dagger$Outcome of the pre-specified
analysis, whose conformal bootstrap let copies of a resampled lesion fall into both halves; it is kept as the
historical record and superseded for inference. Corrected analysis (identity-preserving bootstrap, 500 replicates
of 100 draws; point estimate from 1,000 draws): model A """ + cs_text("A") + r""", model B """ + cs_text("B") + r"""
coverage points; the coverage of these intervals was not validated (Section~\ref{sec:res-conformal}).}
\label{tab:scorecard}
\begin{tabular}{@{}llccl@{}}
\toprule
Test & Hypothesis or criterion & Model & Estimate [95\% CI] & Result \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("Table2_scorecard.tex", body)


# ---------------------------------------------------------------- Table III: balanced accuracy by method
def table3():
    A, B = load("E7_analysis.json"), load("E7_analysis_panderm.json")
    cols = []
    for d in (A, B):
        for metric in ("pooled", "mean_of_centers"):
            for kind in ("image", "lesion"):
                v = dict(zip(d["methods"], d["point"][metric]["mean_ba"][kind]))
                cols.append(v)
    best = [max(c.values()) for c in cols]
    lines = []
    for m in ORDER:
        cells = []
        for c, b in zip(cols, best):
            if m not in c:
                cells.append("--")
            else:
                s = f"{c[m]:.3f}"
                cells.append(r"\textbf{" + s + "}" if abs(c[m] - b) < 1e-12 else s)
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + r" \\")
        if m == "fedper_r9":
            lines.append(r"\midrule")
    body = r"""\begin{table*}[!t]
\centering
\caption{Balanced accuracy on image-level (Img) and lesion-grouped (Les) test sets (E2/E7, mean of five
seeds). Pooled: one balanced accuracy over all test images; mean of clients: average of the six per-client
values. Best value per column in bold; differences of a few points are within seed-to-seed variation
(Supplementary~S9). Model B has no FedBN (no BatchNorm in a linear head); its FedPer is the FedPer-style
two-layer head, whereas its other heads are linear. Kendall's $\tau_b$ compares the rankings of all
methods of each model (eight for model A, seven for model B).}
\label{tab:ba}
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{4}{c}{Model A (EfficientNet-B0, fine-tuned)} & \multicolumn{4}{c}{Model B (PanDerm-Base, frozen)} \\
\cmidrule(lr){2-5}\cmidrule(l){6-9}
 & \multicolumn{2}{c}{Pooled} & \multicolumn{2}{c}{Mean of clients} & \multicolumn{2}{c}{Pooled} & \multicolumn{2}{c}{Mean of clients} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
Method & Img & Les & Img & Les & Img & Les & Img & Les \\
\midrule
""" + "\n".join(lines) + r"""
\midrule
Kendall $\tau_b$ (Img vs Les) & \multicolumn{2}{c}{""" + f"{A['point']['pooled']['tau_between']:+.2f}" + r"""} & \multicolumn{2}{c}{""" \
        + f"{A['point']['mean_of_centers']['tau_between']:+.2f}" + r"""} & \multicolumn{2}{c}{""" \
        + f"{B['point']['pooled']['tau_between']:+.2f}" + r"""} & \multicolumn{2}{c}{""" \
        + f"{B['point']['mean_of_centers']['tau_between']:+.2f}" + r"""} \\
\bottomrule
\end{tabular}
\end{table*}
"""
    write("Table3_balanced_accuracy.tex", body.replace("+-", "-").replace("-0.", "$-$0."))


# ---------------------------------------------------------------- Table IV: conformal methods on held-out-center models
def table4():
    meth = [("global", "Global$^m$"), ("clientwise", "Client-wise"), ("weighted", "Sample-weighted$^m$"), ("fcp", "FCP$^m$"),
            ("shrink", "ShrinkFCP"), ("eb", r"EB-ShrinkFCP$^\dagger$")]
    data = [load("E6_conformal_loco.json")["lac_a0.1"], load("E6_conformal_loco_pandermfrozen.json")["lac_a0.1"]]
    sens = load("Conformal_sensitivity.json") if os.path.exists(os.path.join(R, "Conformal_sensitivity.json")) else None
    if sens is not None and "A/E6_loco" in sens and "B/E6_loco" in sens:     # post hoc comparator, 1,000 draws
        meth.append(("lsw", "Label-shift-weighted$^p$"))
        for d, key in zip(data, ("A/E6_loco", "B/E6_loco")):
            d["lsw"] = sens[key]["mc_1000"]["lsw"]
    cols = []
    for d in data:
        cols.append({k: d[k]["worst_center_q10"] for k, _ in meth})
        cols.append({k: min(d[k]["mean_coverage"]) for k, _ in meth})
        cols.append({k: d[k]["set_size_equal_weight"] for k, _ in meth})
    best = [max(cols[0].values()), max(cols[1].values()), None, max(cols[3].values()), max(cols[4].values()), None]
    lines = []
    for k, lab in meth:
        cells = []
        for c, b in zip(cols, best):
            v = c[k]
            s = f"{v:.2f}" if b is None else f"{v:.3f}"
            cells.append(r"\textbf{" + s + "}" if b is not None and abs(v - b) < 1e-12 else s)
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    body = r"""\begin{table}[!t]
\centering
\caption{Federated conformal calibration on the 18 LOCO models per encoder (LAC, $\alpha=0.1$, 100
lesion-wise calibration/evaluation draws; held-out-client labels enter temperature scaling and conformal
calibration). q10: for each model, the 10th percentile of coverage over draws in its worst client, averaged
over the 18 models (primary; the worst client of the model-averaged q10 is given in Supplementary~S18);
min cov.: lowest client mean coverage; $|C|$: mean set size (equal weight
per client). Best in bold. $^m$Targets coverage on the mixture of clients, not per client.
$^\dagger$Proposed (exploratory). $^p$Post hoc comparator (1,000 draws).}
\label{tab:conformal}
\setlength{\tabcolsep}{2.6pt}
\begin{tabular}{@{}lcccccc@{}}
\toprule
 & \multicolumn{3}{c}{Model A} & \multicolumn{3}{c}{Model B} \\
\cmidrule(lr){2-4}\cmidrule(l){5-7}
Method & q10 & min cov. & $|C|$ & q10 & min cov. & $|C|$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("Table4_conformal.tex", body)


def preview():
    body = r"""\documentclass[10pt,twocolumn]{article}
% IEEE-like page for checking widths: 7.16 in text, 0.17 in column gap (the paper itself uses IEEEtran)
\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=7.16in,textheight=9.5in,columnsep=0.17in]{geometry}
\usepackage{mathptmx}
\usepackage{booktabs,multirow}
\begin{document}
\input{Table1_dataset.tex}
\input{Table2_scorecard.tex}
\input{Table3_balanced_accuracy.tex}
\input{Table4_conformal.tex}
\mbox{}
\end{document}
"""
    write("tables_preview.tex", body)


if __name__ == "__main__":
    table1()
    table2()
    table3()
    table4()
    table5()
    preview()
