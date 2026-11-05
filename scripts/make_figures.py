"""
All main-paper figures, built only from the result files in results/ (no hand-typed numbers).

Format: IEEE (JBHI) column widths - single 3.5 in, double 7.16 in; Liberation Sans / Arial 7-8 pt;
vector PDF with embedded TrueType fonts (IEEE PDF eXpress compatible) plus a 300 dpi PNG of each.
Colours: a validated colour-blind-safe categorical palette (adjacent-pair CVD dE >= 8 on white);
identity is never carried by colour alone (direct labels, markers or legends).

Usage (from the repo folder):  python scripts\\make_figures.py
Writes figures/Fig1_*.pdf/.png ... figures/Fig7_*.pdf/.png
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

R = "results"
OUT = "figures"
SINGLE, DOUBLE = 3.5, 7.16

# ---------------------------------------------------------------- style
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
LESION, IMAGE = BLUE, ORANGE                     # split type, used the same way in every figure
SEQ = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "lines.linewidth": 1.2, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "mathtext.default": "regular",
})

NAMES = {"fedavg_r9": "FedAvg", "fedprox_r9": "FedProx", "scaffold_r9": "SCAFFOLD", "fedbn_r9": "FedBN",
         "fedavgft_r9": "FedAvg-FT", "fedper_r9": "FedPer", "pooled_e20": "Pooled", "local_e20": "Local-only"}


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {OUT}/{name}.pdf / .png")


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def panel(ax, letter, title=None, x=-0.02):
    ax.set_title(f"({letter})" + (f" {title}" if title else ""), x=x, ha="left", fontsize=8)


# ---------------------------------------------------------------- Fig. 1 study design and leakage per center
def fig1():
    pc = load("E7_analysis.json")["point"]["per_center"]
    leak, new = np.array(pc["leakage_rate"]), np.array(pc["new_lesion_images"])
    leaked = new * leak / (1 - leak)
    fig = plt.figure(figsize=(DOUBLE, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.18)

    # (a) schematic: two lanes (image-level, lesion-grouped), train | test columns
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0.1, 6.0)
    ax.axis("off")
    panel(ax, "a", "Why image-level splitting leaks", x=0)
    R_IMG, GAP = 0.19, 0.48
    TRAIN_FC, TEST_EC = "#6b7280", INK

    def group(x0, y, n, label, test=False):
        """A lesion = rounded outline around n images; returns centres of the images."""
        w = (n - 1) * GAP + 2 * R_IMG + 0.22
        ax.add_patch(FancyBboxPatch((x0, y - 0.33), w, 0.66, boxstyle="round,pad=0,rounding_size=0.3",
                                    fc="white", ec=AXIS, lw=0.7, zorder=1))
        xs = [x0 + 0.11 + R_IMG + k * GAP for k in range(n)]
        for xc in xs:
            ax.add_patch(plt.Circle((xc, y), R_IMG, fc="white" if test else TRAIN_FC,
                                    ec=TEST_EC if test else TRAIN_FC, lw=1.0, zorder=2))
        ax.text(x0 + w / 2, y - 0.52, label, ha="center", va="top", fontsize=5.8, color=INK2)
        return xs

    # column headers and divider
    ax.text(4.0, 5.72, "Training set", ha="center", fontsize=7, weight="bold")
    ax.text(8.3, 5.72, "Test set", ha="center", fontsize=7, weight="bold")
    ax.plot([6.75, 6.75], [1.05, 5.45], color=AXIS, lw=0.8, ls=(0, (3, 2)))
    ax.plot([0.05, 9.95], [3.0, 3.0], color=GRID, lw=0.8)

    # lane labels with a colour bar
    for y, col, t1, t2 in ((4.45, IMAGE, "Image-level", "split"), (1.85, LESION, "Lesion-", "grouped")):
        ax.add_patch(plt.Rectangle((0.05, y - 0.7), 0.07, 1.4, fc=col, ec="none"))
        ax.text(0.25, y + 0.05, t1, fontsize=6.8, weight="bold", color=col, va="bottom")
        ax.text(0.25, y + 0.0, t2, fontsize=6.8, weight="bold", color=col, va="top")

    # lane 1: every lesion has images on both sides; test shown in reverse order so links nest
    yy = 4.45
    tr = [group(2.1 + k * 1.45, yy, 2, f"L{k + 1}") for k in range(3)]
    te = {k: group(7.2 + (2 - k) * 0.82, yy, 1, f"L{k + 1}", test=True) for k in range(3)}
    for k in range(3):
        d = te[k][0] - tr[k][-1]
        ax.annotate("", xy=(tr[k][-1] + 0.1, yy + R_IMG + 0.06), xytext=(te[k][0] - 0.02, yy + R_IMG + 0.06),
                    arrowprops=dict(arrowstyle="-|>", color=IMAGE, lw=0.9, mutation_scale=6,
                                    connectionstyle=f"arc3,rad={0.12 + 0.025 * d}", shrinkA=1, shrinkB=1), zorder=3)
    ax.text(8.3, 3.25, "siblings are in training", ha="center", va="center", fontsize=6.1, color=IMAGE)

    # lane 2: lesions kept whole
    yy = 1.85
    group(2.1, yy, 3, "L1")
    group(4.1, yy, 3, "L2")
    group(7.55, yy, 3, "L3", test=True)
    ax.text(8.35, 0.75, "lesions unseen in training", ha="center", va="center", fontsize=6.1, color=LESION)

    # key
    ax.add_patch(plt.Circle((0.35, 0.35), 0.12, fc=TRAIN_FC, ec=TRAIN_FC))
    ax.text(0.55, 0.35, "training image", va="center", fontsize=5.9, color=INK2)
    ax.add_patch(plt.Circle((2.55, 0.35), 0.12, fc="white", ec=TEST_EC, lw=0.9))
    ax.text(2.75, 0.35, "test image", va="center", fontsize=5.9, color=INK2)
    ax.text(4.35, 0.35, "outline = one lesion", va="center", fontsize=5.9, color=INK2)

    # (b) leaked vs new test images per center
    ax = fig.add_subplot(gs[1])
    y = np.arange(6)[::-1]
    ax.barh(y, leaked, color=IMAGE, height=0.62, label="Lesion also in training (leaked)")
    ax.barh(y, new, left=leaked, color=LESION, height=0.62, label="New lesion", edgecolor="white", linewidth=0.8)
    for yi, l, n, r in zip(y, leaked, new, leak):
        ax.text(l + n + 25, yi, f"{100 * r:.0f}%" if r >= 0.01 else f"{100 * r:.1f}%", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Center {c}" for c in range(6)])
    ax.set_xlabel("Image-level test images (mean of 5 splits)")
    ax.set_xlim(0, max(leaked + new) * 1.18)
    xgrid(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", handlelength=1.0, borderaxespad=0.1)
    panel(ax, "b", "Leaked share of each center's test set", x=-0.28)
    save(fig, "Fig1_design_leakage")


# ---------------------------------------------------------------- Fig. 2 leakage inflation (Gate 2)
def fig2():
    rows = []
    for f, model in (("E1_gate2.json", "A: EfficientNet-B0, full fine-tuning"),
                     ("E1_gate2_effb0frozen.json", "A-frozen: EfficientNet-B0, linear head"),
                     ("E1_gate2_pandermfrozen.json", "B: PanDerm-Base, linear head")):
        g = load(f)
        for key, lab in (("fedavg_r9 (PRIMARY)", "FedAvg, 9 rounds"), ("fedavg_r27", "FedAvg, 27 rounds"),
                         ("pooled_e20", "Pooled (centralized)")):
            if key in g:
                v = g[key]
                rows.append((model, lab, 100 * v["inflation"], [100 * c for c in v["ci95"]],
                             [100 * (s["ba_full"] - s["ba_new_reweighted"]) if "ba_new_reweighted" in s else np.nan
                              for s in v["per_seed"]]))
    fig, ax = plt.subplots(figsize=(SINGLE, 2.55))
    ypos, labels, y, prev = [], [], 0, None
    heads = []
    for model, lab, inf, ci, seeds in rows:
        if model != prev:
            y -= 0.55 if prev else 0
            heads.append((y, model))
            y -= 0.75
            prev = model
        primary = lab.startswith("FedAvg, 9")
        mk = "o" if lab.startswith("FedAvg") else "s"
        col = INK if primary else INK2
        ax.plot(ci, [y, y], color=col, lw=1.4 if primary else 1.0, solid_capstyle="round")
        ax.plot(inf, y, mk, ms=5.2 if primary else 4.4, mfc=col if lab.startswith("FedAvg, 9") else "white",
                mec=col, mew=1.1, zorder=3)
        ax.text(ci[1] + 0.6, y, f"{inf:+.1f} [{ci[0]:.1f}, {ci[1]:.1f}]", va="center", fontsize=6.3, color=INK2)
        ypos.append(y)
        labels.append(lab)
        y -= 0.62
    for yh, m in heads:
        ax.text(-0.5, yh, m, fontsize=6.8, weight="bold", va="center", ha="left", transform=ax.get_yaxis_transform())
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(y + 0.2, 0.4)
    ax.set_xlim(-1, 29)
    ax.set_xlabel("Leakage inflation (balanced-accuracy points)\nfull image-level test set minus new-lesion subset")
    xgrid(ax)
    fig.legend(handles=[Line2D([], [], marker="o", ls="-", color=INK, mfc=INK, ms=4.5, label="Primary (pre-specified)"),
                        Line2D([], [], marker="o", ls="-", color=INK2, mfc="white", ms=4, label="27 rounds"),
                        Line2D([], [], marker="s", ls="-", color=INK2, mfc="white", ms=4, label="Pooled")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), handlelength=1.6, columnspacing=1.0)
    save(fig, "Fig2_leakage_inflation")


# ---------------------------------------------------------------- Fig. 3 rankings image vs lesion
def bump(ax, d, metric, title, letter, highlight):
    methods = d["methods"]
    mi = np.array(d["point"][metric]["mean_ba"]["image"])
    ml = np.array(d["point"][metric]["mean_ba"]["lesion"])
    ri = (-mi).argsort().argsort() + 1
    rl = (-ml).argsort().argsort() + 1
    hcols = [RED, VIOLET, GREEN]
    hk = 0
    for j, m in enumerate(methods):
        hl = m in highlight
        col = hcols[hk] if hl else AXIS
        if hl:
            hk += 1
        ax.plot([0, 1], [ri[j], rl[j]], color=col, lw=2.0 if hl else 1.0, zorder=3 if hl else 2, solid_capstyle="round")
        for x, r, v in ((0, ri[j], mi[j]), (1, rl[j], ml[j])):
            ax.plot(x, r, "o", ms=4.6 if hl else 3.6, color=col if hl else MUTED, mec="white", mew=0.8, zorder=4)
        ax.text(-0.07, ri[j], f"{NAMES[m]}", ha="right", va="center", fontsize=6.3,
                color=INK if hl else INK2, weight="bold" if hl else "normal")
        ax.text(1.07, rl[j], f"{NAMES[m]}", ha="left", va="center", fontsize=6.3,
                color=INK if hl else INK2, weight="bold" if hl else "normal")
    n = len(methods)
    ax.set_ylim(n + 0.6, 0.4)
    ax.set_xlim(-0.72, 1.72)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Image-\nlevel", "Lesion-\ngrouped"])
    ax.get_xticklabels()[0].set_color(IMAGE)
    ax.get_xticklabels()[1].set_color(LESION)
    for t in ax.get_xticklabels():
        t.set_weight("bold")
    ax.set_yticks([])
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    p = d["point"][metric]
    ci = d["ci95"][f"{metric}.tau_between"]
    ax.text(0.5, -0.17, f"Kendall τ = {p['tau_between']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]\n"
            f"single-seed τ {p['tau_cross_single_seed']:+.2f} vs noise floor {p['tau_noise_floor']:+.2f}",
            ha="center", va="top", fontsize=6.3, color=INK2, transform=ax.get_xaxis_transform())
    panel(ax, letter, title, x=0.0)


def fig3():
    A, B = load("E7_analysis.json"), load("E7_analysis_panderm.json")
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.75), gridspec_kw={"wspace": 0.22})
    bump(axs[0], A, "pooled", "Model A, pooled BA", "a", ["fedper_r9"])
    bump(axs[1], A, "mean_of_centers", "Model A, mean of centers", "b", ["local_e20"])
    bump(axs[2], B, "pooled", "Model B, pooled BA", "c", ["fedper_r9", "pooled_e20"])
    save(fig, "Fig3_rankings")


# ---------------------------------------------------------------- Fig. 4 mechanism: capacity and proximity
def fig4():
    A, B = load("E7_analysis.json"), load("E7_analysis_panderm.json")
    nn = load("NN_proximity.json")
    fig = plt.figure(figsize=(DOUBLE, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 0.85], wspace=0.55)

    # (a) dose-response, model A
    ax = fig.add_subplot(gs[0])
    pc = A["point"]["per_center"]
    lk = 100 * np.array(pc["leakage_rate"])
    w = np.array(pc["new_lesion_images"])
    show = [("local_e20", RED), ("pooled_e20", AQUA), ("fedper_r9", VIOLET), ("fedavg_r9", BLUE)]
    xs = np.linspace(0, 90, 50)
    for m, col in show:
        y = 100 * np.array(pc["inflation"][m])
        ok = np.isfinite(y) & (w > 0)
        X = np.c_[np.ones(ok.sum()), lk[ok]]
        W = np.diag(w[ok])
        b = np.linalg.solve(X.T @ W @ X, X.T @ W @ y[ok])
        ax.plot(xs, b[0] + b[1] * xs, color=col, lw=1.4)
        ax.scatter(lk[ok], y[ok], s=6 + w[ok] / 18, color=col, edgecolor="white", lw=0.6, zorder=3)
        ci = A["ci95"][f"slope.{m}"]
        ax.text(91, b[0] + b[1] * 90, f"{NAMES[m]}", va="center", fontsize=6.3, color=INK)
    ax.axhline(0, color=AXIS, lw=0.7)
    ax.set_xlim(-3, 120)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("Center leakage rate (% of test images)")
    ax.set_ylabel("Per-center inflation (BA points)")
    ygrid(ax)
    panel(ax, "a", "Dose-response (model A)", x=-0.2)

    # (b) H3b inflation by method, models A and B
    ax = fig.add_subplot(gs[1])
    order = ["fedavg_r9", "fedprox_r9", "scaffold_r9", "fedbn_r9", "fedavgft_r9", "fedper_r9", "pooled_e20", "local_e20"]
    ia = A["point"]["h3b"]["inflation_by_method_centers_023"]
    ib = B["point"]["h3b"]["inflation_by_method_centers_023"]
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        a = 100 * ia[m]
        b = 100 * ib[m] if m in ib else None
        if b is not None:
            ax.plot([a, b], [yi, yi], color=GRID, lw=1.6, zorder=1)
        ax.plot(a, yi, "o", ms=5, color=BLUE, mec="white", mew=0.7, zorder=3)
        if b is not None:
            ax.plot(b, yi, "D", ms=4.4, color=MAGENTA, mec="white", mew=0.7, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES[m] for m in order])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 16)
    ax.set_ylim(-0.6, len(order) + 0.9)
    ax.set_xlabel("Inflation, centers 0, 2, 3 (BA points)")
    xgrid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4.5, label="Model A (full fine-tuning)"),
                       Line2D([], [], marker="D", ls="", color=MAGENTA, ms=4, label="Model B (frozen; FedPer = 256-unit head)")],
              loc="upper right", bbox_to_anchor=(1.03, 1.0), handletextpad=0.3, fontsize=6.2)
    panel(ax, "b", "Inflation follows capacity", x=-0.35)

    # (c) 1-NN vs linear head, frozen encoders
    ax = fig.add_subplot(gs[2])
    encs = [("effb0", "A-frozen"), ("panderm", "B")]
    xb = np.arange(2)
    hv = [100 * nn[e]["mean"]["head_inflation"] for e, _ in encs]
    nv = [100 * nn[e]["mean"]["nn1_inflation"] for e, _ in encs]
    hs = [100 * nn[e]["sd"]["head_inflation"] for e, _ in encs]
    ns = [100 * nn[e]["sd"]["nn1_inflation"] for e, _ in encs]
    ax.bar(xb - 0.19, hv, 0.34, yerr=hs, color=MUTED, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK2), label="Linear head")
    ax.bar(xb + 0.19, nv, 0.34, yerr=ns, color=VIOLET, error_kw=dict(lw=0.7, capsize=1.5, ecolor=INK2), label="1-nearest neighbour")
    for x, v, e in zip(list(xb - 0.19) + list(xb + 0.19), hv + nv, hs + ns):
        ax.text(x, v + e + 0.7, f"{v:.1f}", ha="center", fontsize=6.3)
    ax.set_xticks(xb)
    ax.set_xticklabels([n for _, n in encs])
    ax.set_ylabel("Inflation (BA points)")
    ax.set_ylim(0, 33)
    ygrid(ax)
    ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.0)
    panel(ax, "c", "Proximity (exploratory)", x=-0.35)
    save(fig, "Fig4_mechanism")


# ---------------------------------------------------------------- Fig. 5 calibration
def heat(ax, cal, methods, title, letter, vmax, ylabels=True):
    M = np.array([cal[f"{m}|lesion"]["ece_by_center"] if f"{m}|lesion" in cal else [np.nan] * 6 for m in methods])
    cmap = mpl.colors.LinearSegmentedColormap.from_list("seq", SEQ[1:])
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#f3f3f1", ec="white", lw=1.2))
                ax.text(j, i, "–", ha="center", va="center", fontsize=6, color=MUTED)
                continue
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.8,
                    color="white" if v > vmax * 0.55 else INK)
    ax.set_xticks(range(6))
    ax.set_xticklabels([str(c) for c in range(6)])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([NAMES[m] for m in methods] if ylabels else [])
    ax.set_xlabel("Center")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    panel(ax, letter, title, x=-0.02)
    return im


def fig5():
    ca, cb = load("E7_calibration.json"), load("E7_calibration_pandermfrozen.json")
    ma = ["fedavg_r9", "fedprox_r9", "scaffold_r9", "fedbn_r9", "fedavgft_r9", "fedper_r9", "pooled_e20", "local_e20"]
    fig = plt.figure(figsize=(DOUBLE, 2.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.3, 1], wspace=0.55)
    left = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
    a1, a2, cax = (fig.add_subplot(left[i]) for i in range(3))
    a3 = fig.add_subplot(outer[1])
    vmax = 0.26
    heat(a1, ca, ma, "Model A: ECE per center", "a", vmax)
    im = heat(a2, cb, ma, "Model B (no FedBN)", "b", vmax, ylabels=False)
    cb_ = fig.colorbar(im, cax=cax)
    cb_.outline.set_visible(False)
    cb_.ax.tick_params(labelsize=6, length=2)
    cb_.set_label("ECE", fontsize=6.5)

    # (c) temperature scaling on clean validation vs leaky test
    items = [("A", "fedavg_r9", ca), ("A", "pooled_e20", ca), ("A", "local_e20", ca), ("B", "fedper_r9", cb)]
    lab = {"fedavg_r9": "FedAvg", "pooled_e20": "Pooled", "local_e20": "Local-only", "fedper_r9": "FedPer-style"}
    y0 = 0
    yt, yl = [], []
    for model, m, cal in items:
        for k, (split, col) in enumerate((("image", IMAGE), ("lesion", LESION))):
            yy = y0 - k * 0.42
            r, t = 100 * cal[f"{m}|{split}"]["ece_raw"], 100 * cal[f"{m}|{split}"]["ece"]
            a3.annotate("", xy=(t, yy), xytext=(r, yy),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, mutation_scale=7, shrinkA=2.5, shrinkB=1))
            a3.plot(r, yy, "o", ms=3.6, mfc="white", mec=col, mew=1.0, zorder=3)
        yt.append(y0 - 0.21)
        yl.append(f"{lab[m]}\n(model {model})")
        y0 -= 1.25
    a3.set_yticks(yt)
    a3.set_yticklabels(yl)
    a3.tick_params(axis="y", length=0)
    a3.spines["left"].set_visible(False)
    a3.set_xlabel("ECE ×100: raw (○) → temperature-scaled")
    a3.set_xlim(0, 16)
    xgrid(a3)
    a3.legend(handles=[Line2D([], [], color=IMAGE, lw=1.3, label="Image-level test (leaky)"),
                       Line2D([], [], color=LESION, lw=1.3, label="Lesion-grouped test (clean)")],
              loc="upper center", bbox_to_anchor=(0.3, -0.22), handlelength=1.2, borderaxespad=0)
    panel(a3, "c", "Scaling on clean validation", x=-0.5)
    save(fig, "Fig5_calibration")


# ---------------------------------------------------------------- Fig. 6 federated conformal on unseen-center models
CF = [("global", "Global", MUTED, "o"), ("fcp", "FCP", INK2, "D"), ("clientwise", "Client-wise", AQUA, "s"),
      ("shrink", "ShrinkFCP", MAGENTA, "^"), ("eb", "EB-ShrinkFCP (ours)", VIOLET, "o")]


def fig6():
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (f, title, letter) in zip(axs, (("E6_conformal_loco.json", "Model A", "a"),
                                            ("E6_conformal_loco_pandermfrozen.json", "Model B", "b"))):
        d = load(f)
        r = d["lac_a0.1"]
        off = np.linspace(-0.32, 0.32, len(CF))
        for (k, lab, col, mk), o in zip(CF, off):
            mean, q10 = np.array(r[k]["mean_coverage"]), np.array(r[k]["lower_tail_q10"])
            x = np.arange(6) + o
            ax.vlines(x, q10, mean, color=col, lw=1.3 if k == "eb" else 0.9)
            ax.plot(x, mean, mk, ms=4.8 if k == "eb" else (4.4 if mk == "^" else 3.8), mfc=col, mec="white", mew=0.6, ls="", zorder=3, label=lab)
            ax.plot(x, q10, "_", ms=4, color=col, mew=1.0)
        ax.axhline(0.9, color=INK, lw=0.7, ls=(0, (4, 2)))
        for c in range(5):
            ax.axvline(c + 0.5, color=GRID, lw=0.6)
        ax.set_xticks(range(6))
        ax.set_xticklabels([str(c) for c in range(6)])
        ax.set_xlim(-0.5, 5.5)
        ax.set_xlabel("Held-out-center model, evaluated on center")
        t = d["tests"]["eb"]
        ax.text(0.02, 0.03, f"Pre-specified test (vs client-wise, worst-center q10):\n"
                f"{100 * t['gain']:+.1f} pts [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}], "
                f"{'criterion met' if t['success'] else 'criterion not met'}",
                transform=ax.transAxes, fontsize=6.2, color=INK, weight="bold" if t["success"] else "normal")
        panel(ax, letter, title + " (leave-one-center-out, LAC, α = 0.1)", x=0.0)
    axs[0].set_ylabel("Coverage: mean (marker), 10th percentile (tick)")
    axs[0].set_ylim(0.6, 1.0)
    h, l = axs[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=INK, lw=0.7, ls=(0, (4, 2))))
    l.append("Target 1 − α")
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07), handletextpad=0.2, columnspacing=1.2)
    save(fig, "Fig6_conformal")


# ---------------------------------------------------------------- Fig. 7 domain shift
def fig7():
    ea, eb = load("E3_E6_analysis.json")["e6"], load("E3_E6_analysis_pandermfrozen.json")["e6"]
    pad = load("E6_padufes.json")
    fig = plt.figure(figsize=(DOUBLE, 2.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    for k, (e, col, mk, lab) in enumerate(((ea, BLUE, "o", "Model A"), (eb, MAGENTA, "D", "Model B"))):
        for c in range(6):
            x = c + (-0.14 if k == 0 else 0.14)
            a, u = 100 * e[str(c)]["ba_in_federation"], 100 * e[str(c)]["ba_unseen"]
            ax.annotate("", xy=(x, u), xytext=(x, a),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, mutation_scale=6, shrinkA=2.5, shrinkB=0.5))
            ax.plot(x, a, mk, ms=3.8, mfc="white", mec=col, mew=1.0, zorder=3)
        ax.plot([], [], mk, mfc="white", mec=col, label=f"{lab}: in federation → held out")
    for c in range(5):
        ax.axvline(c + 0.5, color=GRID, lw=0.6)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{c}" for c in range(6)])
    ax.set_xlim(-0.5, 5.5)
    ax.set_xlabel("Center (4, 5: < 200 test images)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_ylim(20, 92)
    ax.set_yticks(range(20, 81, 10))
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, borderaxespad=0.1, columnspacing=0.8, fontsize=6.3)
    panel(ax, "a", "Unseen centers (lesion-grouped, seeds 0–2)", x=-0.12)

    ax = fig.add_subplot(gs[1])
    xs = np.arange(2)
    bars = []
    for k, (m, col) in enumerate((("A", BLUE), ("B", MAGENTA))):
        r = pad[m]["lac_a0.1"]
        isic, tr, rc = r["transfer"]["isic_coverage"], r["transfer"]["coverage"], r["recalibrated"]["coverage"]
        vals = [isic, tr, rc]
        sizes = [None, r["transfer"]["set_size"], r["recalibrated"]["set_size"]]
        xx = np.arange(3) + (-0.18 if k == 0 else 0.18)
        bars.append(ax.bar(xx, vals, 0.34, color=col, label=f"Model {m}"))
        lo, hi = r["transfer"]["coverage_ci95"]
        ax.errorbar(xx[1], tr, yerr=[[tr - lo], [hi - tr]], color=INK2, lw=0.7, capsize=1.5)
        for x, v, s in zip(xx, vals, sizes):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=5.9)
            if s is not None:
                ax.text(x, 0.04, f"|C|\n{s:.1f}", ha="center", fontsize=5.4, color="white", linespacing=0.9)
    ax.axhline(0.9, color=INK, lw=0.7, ls=(0, (4, 2)))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Fed-ISIC2019\n(in domain)", "PAD-UFES-20\ntransferred", "PAD-UFES-20\nrecalibrated"])
    ax.set_ylim(0, 1.17)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Coverage (LAC, α = 0.1)")
    ygrid(ax)
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, borderaxespad=0.1)
    panel(ax, "b", "Smartphone-image shift (PAD-UFES-20)", x=-0.12)
    ax.text(0.99, -0.27, "|C| = mean prediction-set size (of 6 classes); recalibrated on 50% of patients", transform=ax.transAxes, ha="right",
            fontsize=5.9, color=INK2)
    save(fig, "Fig7_domain_shift")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
