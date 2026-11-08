"""
Checks the numbers in the paper against the result files (final check before submission).

1. Key numbers of the abstract and results: each is recomputed from results/*.json, rounded as in the
   text, and searched for in paper/paper_single.tex.
2. Tables and supplement: tables/*.tex and paper/supplement.tex are regenerated from the result files and
   compared with the current versions (they must be identical).

Usage (from the repo folder):  python scripts\\verify_numbers.py
"""
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

R = "results"


def j(name):
    return json.load(open(os.path.join(R, name)))


def pts(x, d=1):
    return f"{100 * x:.{d}f}"


def num(x, d=3):
    return f"{x:.{d}f}"


def claims():
    g2, e8, e9 = j("E1_gate2.json"), j("E8_sibling_exposure.json"), j("E9_analysis_panderm.json")
    e7a, e7b = j("E7_analysis.json"), j("E7_analysis_panderm.json")
    loA, loB = j("E6_conformal_loco.json"), j("E6_conformal_loco_pandermfrozen.json")
    e5B = j("E5_conformal_pandermfrozen.json")
    pad, aud = j("E6_padufes.json"), j("Audit_supplement.json")
    cc, sim, ebv = j("Class_conditional.json"), j("Conformal_simulation.json"), j("EB_variants.json")
    nn, bud = j("NN_matched.json"), j("Budget_controls.json")
    f = e8["families"]
    off = {r["client"]: r["leaked_same_client"] + r["leaked_other_client_only"] for r in aud["leakage_official"]}
    out = [
        ("Official-split leakage, client 1 (%)", pts(off[1]), "0.1"),
        ("Official-split leakage, client 0 (%)", f"{100 * off[0]:.0f}", "93"),
        ("Gate 2: FedAvg inflation", pts(g2["fedavg_r9 (PRIMARY)"]["inflation"]), "8.1"),
        ("Gate 2: CI low", pts(g2["fedavg_r9 (PRIMARY)"]["ci95"][0]), "4.7"),
        ("Gate 2: CI high", pts(g2["fedavg_r9 (PRIMARY)"]["ci95"][1]), "11.2"),
        ("Pooled inflation, model A", pts(g2["pooled_e20"]["inflation"]), "16.2"),
        ("FedAvg 27 rounds inflation", pts(g2["fedavg_r27"]["inflation"]), "12.4"),
        ("E8 FedAvg exposure effect", pts(e8["primary"]["effect"]), "9.1"),
        ("E8 CI low", pts(e8["primary"]["ci"][0]), "5.5"),
        ("E8 CI high", pts(e8["primary"]["ci"][1]), "12.3"),
        ("E8 local-only model A", pts(f["local_e20"]["effect"]["ba"]), "28.8"),
        ("E8 smallest effect (model B pooled)", pts(f["pandermfrozen_pooled_e20"]["effect"]["ba"]), "4.5"),
        ("Model B tau_b", num(e7b["point"]["pooled"]["tau_between"], 2), "0.05"),
        ("Model A tau_b", num(e7a["point"]["pooled"]["tau_between"], 2), "0.79"),
        ("H3b model A difference", pts(e7a["point"]["h3b"]["difference"]), "3.2"),
        ("E9 tau_b with two-layer heads", num(e9["point"]["pooled"]["tau_between"], 2), "0.20"),
        ("Global threshold, lowest client mean (model B, E5)", num(min(e5B["lac_a0.1"]["global"]["mean_coverage"]), 2), "0.72"),
        ("Global threshold, lowest client mean (model A, LOCO)", num(min(loA["lac_a0.1"]["global"]["mean_coverage"]), 2), "0.83"),
        ("EB gain model A (5b)", pts(loA["tests"]["eb"]["gain"]), "2.3"),
        ("EB gain CI low", pts(loA["tests"]["eb"]["ci95"][0]), "0.8"),
        ("EB gain CI high", pts(loA["tests"]["eb"]["ci95"][1]), "3.9"),
        ("EB gain model B (5b)", pts(loB["tests"]["eb"]["gain"]), "0.5"),
        ("PAD transferred coverage A", num(pad["A"]["lac_a0.1"]["transfer"]["coverage"], 2), "0.60"),
        ("PAD transferred coverage B", num(pad["B"]["lac_a0.1"]["transfer"]["coverage"], 2), "0.67"),
        ("Class-conditional: DF coverage, model A LOCO, client-wise", num(cc["isic"]["A/E6_loco"]["clientwise"]["min_class_cov"], 2), "0.58"),
        ("Class-conditional: Mondrian set size, A LOCO", num(cc["isic"]["A/E6_loco"]["mondrian_pooled"]["size"], 1), "3.5"),
        ("Simulation: variance ratio max", num(max(sim["variance_check"]["rho0.2"]["ratio_approx_over_true"]), 1), "2.6"),
        ("Simulation: variance ratio min", num(min(sim["variance_check"]["rho0.6"]["ratio_approx_over_true"]), 1), "1.5"),
        ("EB variants: random-effects mean, largest drop (model B E5)",
         pts(ebv["B/E5"]["prespecified"]["worst_center_q10"] - ebv["B/E5"]["re_mean"]["worst_center_q10"]), "2.2"),
        ("NN matched proximity A-frozen", num(nn["effb0"]["mean"]["sim_matched_diff"], 3), "0.059"),
        ("NN matched proximity model B", num(nn["panderm"]["mean"]["sim_matched_diff"], 3), "0.055"),
        ("1-NN inflation with siblings, A-frozen", pts(nn["effb0"]["mean"]["nn1_std_inflation"]), "25.9"),
        ("1-NN inflation with siblings, model B", pts(nn["panderm"]["mean"]["nn1_std_inflation"]), "25.2"),
        ("Budget: FedAvg-FT 1 step, lesion BA model A", num(bud["A"]["fedavgft1_r9"]["ba_lesion"]), "0.480"),
        ("Budget: FedAvg-FT 30 steps, lesion BA model A", num(bud["A"]["fedavgft30_r9"]["ba_lesion"]), "0.564"),
        ("Budget: local 60 epochs inflation, A-frozen", pts(bud["A-frozen"]["local_e60"]["inflation"]), "7.7"),
        ("Budget: pooled 60 epochs inflation, model B", pts(bud["B"]["pooled_e60"]["inflation"]), "6.3"),
    ]
    return out


def main():
    tex = open(os.path.join("paper", "paper_single.tex"), encoding="utf-8").read()
    flat = re.sub(r"\s+", " ", tex)
    bad = 0
    print("=== 1. Key numbers: result file vs paper text ===")
    for label, computed, in_text in claims():
        same = computed == in_text
        found = re.search(r"(?<![\d.])" + re.escape(in_text) + r"(?![\d])", flat) is not None
        ok = same and found
        bad += not ok
        why = "" if ok else (f"  <- result file gives {computed}, text has {in_text}" if not same else "  <- not found in paper text")
        print(f"  {'OK  ' if ok else 'FAIL'} {label:62s} {computed:>6s}{why}")

    print("=== 2. Tables and supplement regenerated from the result files ===")
    tmp = tempfile.mkdtemp()
    for p in ["tables", os.path.join("paper", "supplement.tex")]:
        dst = os.path.join(tmp, os.path.basename(p))
        shutil.copytree(p, dst) if os.path.isdir(p) else shutil.copy(p, dst)
    for s in ("make_tables.py", "make_supplement.py"):
        r = subprocess.run([sys.executable, os.path.join("scripts", s)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAIL {s}: {r.stderr.strip()[-300:]}")
            bad += 1
    pairs = [(os.path.join(tmp, "tables", f), os.path.join("tables", f)) for f in os.listdir(os.path.join(tmp, "tables"))
             if f.endswith(".tex")]
    pairs.append((os.path.join(tmp, "supplement.tex"), os.path.join("paper", "supplement.tex")))
    for old, new in pairs:
        a = open(old, encoding="utf-8").read().replace("\r", "")
        b = open(new, encoding="utf-8").read().replace("\r", "")
        ok = a == b
        bad += not ok
        print(f"  {'OK  ' if ok else 'FAIL'} {os.path.relpath(new)}{'' if ok else '  <- differs from the version regenerated now'}")
    print()
    print("ALL NUMBERS MATCH" if bad == 0 else f"{bad} CHECK(S) FAILED - send this output")


if __name__ == "__main__":
    main()
