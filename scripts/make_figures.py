"""
All main-paper figures, built only from the result files in results/ (no hand-typed numbers).

Format: IEEE (JBHI) column widths - single 3.5 in, double 7.16 in; Liberation Sans / Arial 7-8 pt;
vector PDF with embedded TrueType fonts (IEEE PDF eXpress compatible) plus a 300 dpi PNG of each.
Colours: a validated colour-blind-safe categorical palette (adjacent-pair CVD dE >= 8 on white);
identity is never carried by colour alone (direct labels, markers or legends).

Usage (from the repo folder):  python scripts\\make_figures.py
Writes figures/Fig1_*.pdf ... figures/Fig8_*.pdf and figures/FigS1_*.pdf (PDF only)
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    "axes.titlelocation": "left", "mathtext.fontset": "custom", "mathtext.rm": "sans",
    "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
})

NAMES = {"fedavg_r9": "FedAvg", "fedprox_r9": "FedProx", "scaffold_r9": "SCAFFOLD", "fedbn_r9": "FedBN",
         "fedavgft_r9": "FedAvg-FT", "fedper_r9": "FedPer", "pooled_e20": "Pooled", "local_e20": "Local-only"}


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))          # vector PDF only (fonts embedded, type 42)
    plt.close(fig)
    print(f"saved {OUT}/{name}.pdf")


def ygrid(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def xgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def panel(ax, letter, title=None, x=-0.02):
    ax.set_title(f"({letter})" + (f" {title}" if title else ""), x=x, ha="left", fontsize=8)


# ---------------------------------------------------------------- dermoscopy thumbnails for the schematics
# Real Fed-ISIC2019 images (ISIC 2019, CC-BY-NC 4.0) are read from the preprocessed image folder. If an image
# is not on disk (e.g. on a machine without the data), a neutral synthetic placeholder is drawn instead, so
# the layout can still be checked; the paper version is made on the machine with the data.
IMG_DIR = os.path.join("data", "fedisic", "ISIC_2019_Training_Input_preprocessed")
SPLIT_FOR_SCHEMATIC = os.path.join("splits", "split_official.csv")   # Fig. 1a shows a lesion leaked in the official split
SIBLING_LESION = os.environ.get("FIG_LESION", "BCN_0000057")      # one lesion with three images (melanoma)
OTHER_LESIONS = ["BCN_0000002", "BCN_0000008"]                      # two other lesions for the training set
_PLACEHOLDER_WARNED = []


def _placeholder(seed, size=256):
    rng = np.random.default_rng(abs(hash(seed)) % (2 ** 32))
    yy, xx = np.mgrid[0:size, 0:size] / size - 0.5
    skin = np.array([0.86, 0.70, 0.62]) + rng.normal(0, 0.02, 3)
    img = np.ones((size, size, 3)) * skin
    cx, cy = rng.normal(0, 0.05, 2)
    ang = rng.uniform(0, np.pi)
    r = np.sqrt(((xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)) ** 2 / 1.0
                + ((xx - cx) * np.sin(ang) - (yy - cy) * np.cos(ang)) ** 2 / 0.6)
    blob = np.clip(1 - r / rng.uniform(0.22, 0.32), 0, 1) ** 0.6
    col = np.array([0.36, 0.22, 0.16]) + rng.normal(0, 0.03, 3)
    img = img * (1 - blob[..., None]) + col * blob[..., None]
    img += rng.normal(0, 0.025, img.shape)
    vign = np.clip(1.25 - 1.6 * np.sqrt(xx ** 2 + yy ** 2), 0, 1)[..., None]
    return np.clip(img * vign, 0, 1)


def lesion_img(name, size=256):
    path = os.path.join(IMG_DIR, f"{name}.jpg")
    if not os.path.exists(path):
        if not _PLACEHOLDER_WARNED:
            print(f"  note: {IMG_DIR} not found; schematic thumbnails are placeholders")
            _PLACEHOLDER_WARNED.append(1)
        return _placeholder(name, size)
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    m = min(w, h)
    im = im.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize((size, size), Image.LANCZOS)
    return np.asarray(im) / 255.0


def thumb(ax, img, x, y, size, ec=None, lw=1.2, r=0.9, z=3):
    """Square image with rounded corners, lower-left corner at (x, y), in data units (axes must be equal-aspect)."""
    im = ax.imshow(img, extent=(x, x + size, y, y + size), zorder=z, interpolation="lanczos")
    clip = FancyBboxPatch((x, y), size, size, boxstyle=f"round,pad=0,rounding_size={r}", transform=ax.transData,
                          fc="none", ec="none")
    im.set_clip_path(clip)
    if ec:
        ax.add_patch(FancyBboxPatch((x, y), size, size, boxstyle=f"round,pad=0,rounding_size={r}", fc="none",
                                    ec=ec, lw=lw, zorder=z + 1))


def canvas(width, height):
    """Figure with one full-size equal-aspect axes in units of 1% of the figure width."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * height / width)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, fc="white", ec=AXIS, lw=0.7, r=1.2, z=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                zorder=z, ls=ls))


def schematic_images():
    sp = pd.read_csv(SPLIT_FOR_SCHEMATIC)
    g = sp[sp.lesion_id == SIBLING_LESION]
    sib = (g[g.fold == "train"].image.tolist() + g[g.fold == "test"].image.tolist())[:3]   # images 1-2 train, 3 test
    print(f"  Fig. 1a lesion {SIBLING_LESION}: " + ", ".join(f"{i + 1}={n}" for i, n in enumerate(sib)))
    others = [sp[sp.lesion_id == l].image.tolist()[:2] for l in OTHER_LESIONS]
    per_center = []
    for c in range(6):
        g = sp[(sp.center == c) & (sp.target == 1)]                     # a nevus from every center
        per_center.append((g.image.iloc[len(g) // 3] if len(g) else sp[sp.center == c].image.iloc[0],
                           int((sp.center == c).sum())))
    return sib, others, per_center


# ---------------------------------------------------------------- Fig. 1 leakage: real sibling images + per-center share
def fig1():
    sib, others, _ = schematic_images()
    S = [lesion_img(n) for n in sib]
    O = [lesion_img(n) for o in others for n in o]
    Hin = 2.75
    fig, ax = canvas(DOUBLE, Hin)
    fig.text(0.004, 0.99, "(a) One lesion, three images: image-level vs lesion-grouped splitting", fontsize=8,
             weight="bold", va="top")

    # source lesion: three images of the same melanoma
    rbox(ax, 0.4, 1.0, 12.0, 31.0, fc="#f6f6f4", ec="none")
    ax.text(6.4, 30.4, "Lesion L", ha="center", fontsize=6.8, weight="bold", va="top")
    ax.text(6.4, 28.2, "3 images of one\nmelanoma", ha="center", fontsize=5.4, color=INK2, va="top", linespacing=1.05)
    for k in range(3):
        y = 16.6 - k * 7.6
        thumb(ax, S[k], 2.8, y, 6.8)
        ax.text(3.9, y + 5.7, f"{k + 1}", fontsize=5.2, color="white", weight="bold", ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.18", fc=INK, ec="none"))

    def lane(y0, col, title, sub):
        ax.add_patch(plt.Rectangle((14.3, y0), 0.45, 14.6, fc=col, ec="none"))
        ax.text(15.6, y0 + 14.5, title, fontsize=6.8, weight="bold", color=col, va="top")
        ax.text(15.6, y0 + 12.3, sub, fontsize=5.5, color=INK2, va="top")
        rbox(ax, 15.6, y0, 22.4, 9.6, fc="#f6f6f4", ec=AXIS)
        rbox(ax, 40.0, y0, 19.6, 9.6, fc="white", ec=col, lw=0.9)
        ax.text(16.4, y0 + 8.9, "train", fontsize=5.5, color=INK2, va="top")
        ax.text(40.8, y0 + 8.9, "test", fontsize=5.5, color=col, va="top", weight="bold")

    # lane 1: image-level split
    y1 = 17.4
    lane(y1, IMAGE, "Image-level split", "images are assigned independently")
    def badge(x, y, t, col=INK):
        ax.text(x, y, t, fontsize=5.2, color="white", weight="bold", ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.18", fc=col, ec="none"))

    for k, x in enumerate((22.4, 30.0)):
        thumb(ax, S[k], x, y1 + 1.2, 6.4)
        badge(x + 1.1, y1 + 6.5, f"{k + 1}")
    thumb(ax, S[2], 42.4, y1 + 1.2, 6.4, ec=IMAGE, lw=1.4)
    badge(43.5, y1 + 6.5, "3", IMAGE)
    ax.annotate("", xy=(36.7, y1 + 3.6), xytext=(42.2, y1 + 3.6),
                arrowprops=dict(arrowstyle="-|>", color=IMAGE, lw=1.2, mutation_scale=7), zorder=6)
    ax.text(51.2, y1 + 6.2, "siblings 1, 2\nare in training", fontsize=5.5, color=IMAGE, va="center", linespacing=1.1)

    # lane 2: lesion-grouped split
    y2 = 1.0
    lane(y2, LESION, "Lesion-grouped split (this study)", "all images of a lesion stay together")
    thumb(ax, O[0], 22.4, y2 + 1.2, 6.4)
    thumb(ax, O[2], 30.0, y2 + 1.2, 6.4)
    ax.text(37.6, y2 + 8.9, "other lesions", fontsize=5.3, color=MUTED, va="top", ha="right")
    for k in range(3):
        thumb(ax, S[k], 41.2 + k * 5.4, y2 + 1.6, 4.9, ec=LESION, lw=1.1)
    ax.text(58.9, y2 + 8.9, "lesion L: 1, 2, 3", fontsize=5.3, color=LESION, va="top", ha="right")

    # (b) leaked vs new test images per center
    pc = load("E7_analysis.json")["point"]["per_center"]
    leak, new = np.array(pc["leakage_rate"]), np.array(pc["new_lesion_images"])
    leaked = new * leak / (1 - leak)
    bx = fig.add_axes([0.715, 0.16, 0.27, 0.72])
    y = np.arange(6)[::-1]
    bx.barh(y, leaked, color=IMAGE, height=0.62, label="Lesion also in training")
    bx.barh(y, new, left=leaked, color=LESION, height=0.62, label="New lesion", edgecolor="white", linewidth=0.8)
    for yi, l, n, r in zip(y, leaked, new, leak):
        bx.text(l + n + 40, yi, f"{100 * r:.0f}%" if r >= 0.01 else f"{100 * r:.1f}%", va="center", fontsize=6.3)
    bx.set_yticks(y)
    bx.set_yticklabels([f"Client {c}" for c in range(6)])
    bx.set_xlabel("Image-level test images (mean of 5 splits)")
    bx.set_xlim(0, max(leaked + new) * 1.2)
    xgrid(bx)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.legend(loc="lower right", handlelength=1.0, borderaxespad=0.1)
    fig.text(0.635, 0.99, "(b) Leaked share of each client's test set", fontsize=8, weight="bold", va="top")
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
    ax.set_xlabel("Apparent inflation (balanced-accuracy points)\nfull image-level test set minus standardized new-lesion subset")
    xgrid(ax)
    fig.legend(handles=[Line2D([], [], marker="o", ls="-", color=INK, mfc=INK, ms=4.5, label="Primary (pre-specified)"),
                        Line2D([], [], marker="o", ls="-", color=INK2, mfc="white", ms=4, label="27 rounds"),
                        Line2D([], [], marker="s", ls="-", color=INK2, mfc="white", ms=4, label="Pooled")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05), handlelength=1.6, columnspacing=1.0)
    save(fig, "Fig3_leakage_inflation")


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
    bump(axs[1], A, "mean_of_centers", "Model A, mean of clients", "b", ["local_e20"])
    bump(axs[2], B, "pooled", "Model B, pooled BA", "c", ["fedper_r9", "pooled_e20"])
    save(fig, "Fig4_rankings")


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
    ax.set_xlabel("Client leakage rate (% of test images)")
    ax.set_ylabel("Per-client inflation (BA points)")
    ygrid(ax)
    panel(ax, "a", "Inflation vs leakage rate (model A)", x=-0.2)

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
    ax.set_xlabel("Inflation, clients 0, 2, 3 (BA points)")
    xgrid(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4.5, label="Model A (full fine-tuning)"),
                       Line2D([], [], marker="D", ls="", color=MAGENTA, ms=4, label="Model B (frozen; FedPer = 256-unit head)")],
              loc="upper right", bbox_to_anchor=(1.03, 1.0), handletextpad=0.3, fontsize=6.2)
    panel(ax, "b", "Inflation by method (H3b clients)", x=-0.35)

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
    save(fig, "Fig5_mechanism")


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
    ax.set_xlabel("Client")
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
    heat(a1, ca, ma, "Model A: ECE per client", "a", vmax)
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
    save(fig, "Fig6_calibration")


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
        ax.set_xlabel("Held-out-client model, evaluated on client")
        t = d["tests"]["eb"]
        ax.text(0.02, 0.03, f"Pre-specified test (vs client-wise, worst-client q10):\n"
                f"{100 * t['gain']:+.1f} pts [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}], "
                f"{'criterion met' if t['success'] else 'criterion not met'}",
                transform=ax.transAxes, fontsize=6.2, color=INK, weight="bold" if t["success"] else "normal")
        panel(ax, letter, title + " (leave-one-client-out, LAC, α = 0.1)", x=0.0)
    axs[0].set_ylabel("Coverage: mean (marker), 10th percentile (tick)")
    axs[0].set_ylim(0.6, 1.0)
    h, l = axs[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=INK, lw=0.7, ls=(0, (4, 2))))
    l.append("Target 1 − α")
    fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.07), handletextpad=0.2, columnspacing=1.2)
    save(fig, "Fig7_conformal")


# ---------------------------------------------------------------- Fig. 7 domain shift
def fig7():
    ea, eb = load("E3_E6_analysis.json")["e6"], load("E3_E6_analysis_pandermfrozen.json")["e6"]
    pad = load("E6_padufes.json")
    fig = plt.figure(figsize=(DOUBLE, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.42, 1.0, 0.95], wspace=0.34)

    # (a) what the shift looks like: one dermoscopy image and one smartphone image of the same class
    ia = fig.add_subplot(gs[0])
    ia.set_xlim(0, 10)
    ia.set_ylim(0, 26)
    ia.set_aspect("equal")
    ia.axis("off")
    sp = pd.read_csv(SPLIT_FOR_SCHEMATIC)
    der = sp[(sp.center == 0) & (sp.target == 2)].image.iloc[5]          # a BCC, Fed-ISIC2019
    pad_csv = os.path.join("splits", "padufes20.csv")
    pad_img = None
    if os.path.exists(pad_csv):
        pdf = pd.read_csv(pad_csv)
        pad_name = pdf[pdf.diagnostic == "BCC"].image.iloc[5]
        pth = os.path.join("data", "padufes20", "preprocessed", f"{pad_name}.jpg")
        if os.path.exists(pth):
            from PIL import Image
            im = Image.open(pth).convert("RGB")
            w, h = im.size
            m = min(w, h)
            pad_img = np.asarray(im.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2)).resize((256, 256))) / 255.0
    if pad_img is None:
        pad_img = _placeholder("pad-ufes")
    thumb(ia, lesion_img(der), 0.5, 16.0, 9.0, r=0.6)
    ia.text(5.0, 15.6, "Fed-ISIC2019\ndermoscopy", ha="center", va="top", fontsize=5.8, linespacing=1.05)
    thumb(ia, pad_img, 0.5, 3.4, 9.0, r=0.6)
    ia.text(5.0, 3.0, "PAD-UFES-20\nsmartphone", ha="center", va="top", fontsize=5.8, linespacing=1.05)
    ia.set_title("(a) Shift (BCC)", x=0.0, ha="left", fontsize=8)

    ax = fig.add_subplot(gs[1])
    for k, (e, col, mk, lab) in enumerate(((ea, BLUE, "o", "Model A"), (eb, MAGENTA, "D", "Model B"))):
        for c in range(6):
            x = c + (-0.14 if k == 0 else 0.14)
            a, u = 100 * e[str(c)]["ba_in_federation"], 100 * e[str(c)]["ba_unseen"]
            ax.annotate("", xy=(x, u), xytext=(x, a),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, mutation_scale=6, shrinkA=2.5, shrinkB=0.5))
            ax.plot(x, a, mk, ms=3.8, mfc="white", mec=col, mew=1.0, zorder=3)
        ax.plot([], [], mk, mfc="white", mec=col, label=lab)
    for c in range(5):
        ax.axvline(c + 0.5, color=GRID, lw=0.6)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"{c}" for c in range(6)])
    ax.set_xlim(-0.5, 5.5)
    ax.set_xlabel("Client (4, 5: < 200 test images)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_ylim(20, 92)
    ax.set_yticks(range(20, 81, 10))
    ax.plot([], [], " ", label="in federation → held out")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, ["Model A", "Model B", "(○ in federation → held out)"], loc="upper right", ncol=1, handlelength=1.0,
              borderaxespad=0.1, fontsize=6.0)
    panel(ax, "b", "Unseen clients (seeds 0–2)", x=-0.12)

    ax = fig.add_subplot(gs[2])
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
    ax.set_xticklabels(["Fed-ISIC\n(calibration)", "PAD\ntransferred", "PAD\nrecalibrated"])
    ax.set_ylim(0, 1.17)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Coverage (LAC, α = 0.1)")
    ygrid(ax)
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, borderaxespad=0.1)
    panel(ax, "c", "Conformal coverage on PAD-UFES-20", x=-0.12)
    ax.text(0.99, -0.27, "|C| = mean prediction-set size (of 6 classes); recalibrated on 50% of patients", transform=ax.transAxes, ha="right",
            fontsize=5.9, color=INK2)
    save(fig, "Fig8_domain_shift")


# ---------------------------------------------------------------- Fig. 2 study framework (with real thumbnails)
def fig_framework():
    sib, _, per_center = schematic_images()
    S = [lesion_img(n) for n in sib]
    W, Hin = DOUBLE, 3.45
    fig, ax = canvas(W, Hin)
    H = 100 * Hin / W                                  # about 48 units
    SOFT, PALE_V = "#f6f6f4", "#f1effc"
    BOT = 8.6                                           # bottom of the four stage panels

    def stage(x, w, n, title):
        rbox(ax, x, BOT, w, H - BOT - 0.4, fc=SOFT, ec="none", r=1.4)
        ax.add_patch(plt.Circle((x + 2.3, H - 3.0), 1.3, fc=INK, ec="none", zorder=3))
        ax.text(x + 2.3, H - 3.05, n, ha="center", va="center", fontsize=6.3, color="white", weight="bold", zorder=4)
        ax.text(x + 4.3, H - 3.0, title, ha="left", va="center", fontsize=7.1, weight="bold")

    def flow(x0, x1, y=27.0):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, mutation_scale=9), zorder=5)

    # 1 six centers
    stage(0.3, 15.8, "1", "Six clients")
    for c, (img, n) in enumerate(per_center):
        yy = H - 11.0 - c * 5.7
        thumb(ax, lesion_img(img), 1.5, yy, 4.8, r=0.6)
        ax.text(7.2, yy + 3.2, f"Client {c}", fontsize=5.9, weight="bold", va="center")
        ax.text(7.2, yy + 1.3, f"{n:,} images", fontsize=5.3, color=INK2, va="center")
    flow(16.3, 18.0)

    # 2 splits
    stage(18.2, 17.6, "2", "Two split types")
    for k, (yy, col, t, sub) in enumerate(((H - 19.6, IMAGE, "Image-level", "siblings cross the split"),
                                          (H - 33.8, LESION, "Lesion-grouped", "lesions kept whole"))):
        rbox(ax, 19.3, yy, 15.4, 12.6, fc="white", ec=col, lw=0.9)
        ax.text(27.0, yy + 11.6, t, ha="center", fontsize=6.4, weight="bold", color=col, va="top")
        ax.text(27.0, yy + 9.3, sub, ha="center", fontsize=5.3, color=INK2, va="top")
        ax.plot([26.9, 26.9], [yy + 0.8, yy + 6.9], color=AXIS, lw=0.7, ls=(0, (2, 1.5)))
        ax.text(23.1, yy + 0.5, "train", ha="center", fontsize=4.9, color=MUTED, va="bottom")
        ax.text(30.8, yy + 0.5, "test", ha="center", fontsize=4.9, color=MUTED, va="bottom")
        if k == 0:
            thumb(ax, S[0], 20.1, yy + 2.5, 3.0, r=0.4)
            thumb(ax, S[1], 23.3, yy + 2.5, 3.0, r=0.4)
            thumb(ax, S[2], 29.3, yy + 2.5, 3.0, ec=IMAGE, lw=1.0, r=0.4)
        else:
            for j in range(3):
                thumb(ax, S[j], 27.3 + j * 2.35, yy + 2.8, 2.2, ec=LESION, lw=0.7, r=0.3)
    ax.text(27.0, BOT + 0.9, "3–5 random re-splits each", ha="center", fontsize=5.3, color=MUTED, va="bottom")
    flow(36.0, 37.7)

    # 3 federated training: server and six clients
    stage(37.9, 25.6, "3", "Federated training")
    sx, sy = 50.7, H - 9.2
    rbox(ax, sx - 6.2, sy - 2.2, 12.4, 4.4, fc=INK, ec="none", r=1.0, z=3)
    ax.text(sx, sy, "Server: aggregate", ha="center", va="center", fontsize=5.8, color="white", weight="bold", zorder=4)
    cxs = np.linspace(40.4, 61.0, 6)
    cy = H - 20.2
    for c, x in enumerate(cxs):
        ax.annotate("", xy=(sx + (x - sx) * 0.18, sy - 2.3), xytext=(x, cy + 3.7),
                    arrowprops=dict(arrowstyle="<|-|>", color=AXIS, lw=0.7, mutation_scale=4), zorder=1)
        rbox(ax, x - 1.6, cy, 3.2, 3.2, fc="white", ec=INK2, lw=0.8, r=0.6, z=3)
        ax.text(x, cy + 1.6, f"{c}", ha="center", va="center", fontsize=5.8, weight="bold", zorder=4)
    ax.text(sx, cy - 0.7, "clients = acquisition sources (images stay local)", ha="center", va="top", fontsize=5.0, color=MUTED)
    ax.text(sx, cy + 5.8, "updates ↑   model ↓", ha="center", fontsize=5.0, color=MUTED, va="center", zorder=4,
            bbox=dict(fc=SOFT, ec="none", pad=0.5))
    ax.text(39.2, H - 24.6, "Encoders", fontsize=6.1, weight="bold", va="top")
    enc = [("A", "EfficientNet-B0, fine-tuned"), ("A-frozen", "EfficientNet-B0, frozen"), ("B", "PanDerm-Base, frozen")]
    for k, (t, d) in enumerate(enc):
        ax.text(39.2, H - 26.8 - k * 2.0, t, fontsize=5.4, weight="bold", va="top")
        ax.text(45.6, H - 26.8 - k * 2.0, d, fontsize=5.4, color=INK2, va="top")
    ax.text(39.2, H - 33.6, "Methods", fontsize=6.1, weight="bold", va="top")
    ax.text(39.2, H - 35.8, "FedAvg · FedProx · SCAFFOLD · FedBN\nFedAvg-FT · FedPer · pooled · local-only",
            fontsize=5.3, color=INK2, va="top", linespacing=1.3)
    flow(63.7, 65.4)

    # 4 evaluation: four question tiles with small glyphs
    stage(65.6, 34.0, "4", "Reliability evaluation")
    tiles = [("RQ2", "Leakage inflation", "full minus new-lesion BA"), ("RQ1", "Calibration", "ECE per client"),
             ("RQ1 · RQ4", "Conformal coverage", "EB-ShrinkFCP (proposed)"), ("RQ3", "Personalization", "H3b; per-client change")]
    tw, th = 15.7, 15.4
    for k, (rq, t, sub) in enumerate(tiles):
        x0 = 66.6 + (k % 2) * (tw + 0.8)
        y0 = H - 6.4 - th - (k // 2) * (th + 0.9)
        hl = k == 2
        rbox(ax, x0, y0, tw, th, fc=PALE_V if hl else "white", ec=VIOLET if hl else AXIS, lw=1.0 if hl else 0.7)
        ax.text(x0 + 1.0, y0 + th - 0.9, rq, fontsize=5.0, color=VIOLET if hl else MUTED, va="top")
        ax.text(x0 + 1.0, y0 + th - 2.9, t, fontsize=6.1, weight="bold", va="top")
        ax.text(x0 + 1.0, y0 + 0.9, sub, fontsize=5.0, color=VIOLET if hl else INK2, va="bottom",
                weight="bold" if hl else "normal")
        gx, gy, gh = x0 + 1.2, y0 + 3.6, 6.2                     # glyph area
        if k == 0:                                    # two bars: full vs new-lesion test set
            for j, (v, col, lab) in enumerate(((0.95, IMAGE, "full"), (0.62, LESION, "new"))):
                bx = gx + 1.0 + j * 5.2
                ax.add_patch(plt.Rectangle((bx, gy), 3.6, gh * v, fc=col, ec="none"))
                ax.text(bx + 1.8, gy - 0.25, lab, ha="center", va="top", fontsize=4.7, color=INK2)
            ax.annotate("", xy=(gx + 10.6, gy + gh * 0.62), xytext=(gx + 10.6, gy + gh * 0.95),
                        arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7, mutation_scale=5))
            ax.text(gx + 11.3, gy + gh * 0.78, "Δ", fontsize=6.0, va="center", color=INK)
        elif k == 1:                                  # reliability diagram
            ax.add_patch(plt.Rectangle((gx, gy), gh, gh, fc="white", ec=AXIS, lw=0.6))
            ax.plot([gx, gx + gh], [gy, gy + gh], color=MUTED, lw=0.7, ls=(0, (2, 1.5)))
            xs = np.linspace(0, 1, 6)
            ax.step(gx + xs * gh, gy + xs ** 1.6 * gh, where="mid", color=BLUE, lw=1.1)
            ax.text(gx + gh + 0.9, gy + gh, "accuracy vs\nconfidence", fontsize=4.7, color=INK2, va="top",
                    linespacing=1.1)
        elif k == 2:                                  # an image with its prediction set
            gy, gh = gy + 1.4, gh - 1.2
            probs = [0.46, 0.31, 0.09, 0.06, 0.04, 0.02, 0.01, 0.01]
            labs = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
            thr = 0.2
            for j, (pv, lab) in enumerate(zip(probs, labs)):
                xx = gx + 0.2 + j * 1.6
                ax.add_patch(plt.Rectangle((xx, gy), 1.15, gh * pv / 0.5, fc=VIOLET if pv >= thr else AXIS, ec="none"))
                ax.text(xx + 0.58, gy - 0.25, lab, ha="center", va="top", fontsize=3.6, color=INK2, rotation=90)
            ax.plot([gx, gx + 12.9], [gy + gh * thr / 0.5] * 2, color=INK, lw=0.7, ls=(0, (2, 1.5)))
            ax.text(gx + 13.0, gy + gh * thr / 0.5, r"$1{-}\hat{q}_c$", fontsize=5.2, va="center")

        else:                                         # per-center dots: global vs personalized
            for c in range(6):
                xx = gx + 0.8 + c * 1.6
                g, p = 0.30 + 0.10 * np.sin(c * 1.7), 0.62 + 0.14 * np.cos(c * 1.3)
                ax.plot([xx, xx], [gy + g * gh, gy + p * gh], color=AXIS, lw=0.8)
                ax.plot(xx, gy + g * gh, "o", ms=2.1, color=MUTED)
                ax.plot(xx, gy + p * gh, "o", ms=2.3, color=BLUE)
            ax.plot(gx + 10.3, gy + gh * 0.85, "o", ms=2.3, color=BLUE)
            ax.text(gx + 11.0, gy + gh * 0.85, "personal", fontsize=4.6, color=INK2, va="center")
            ax.plot(gx + 10.3, gy + gh * 0.55, "o", ms=2.1, color=MUTED)
            ax.text(gx + 11.0, gy + gh * 0.55, "global", fontsize=4.6, color=INK2, va="center")

    # bottom ribbon: stress tests and pre-registration
    rbox(ax, 0.3, 0.4, 99.4, 7.0, fc="white", ec=AXIS, lw=0.6, r=1.2)
    ax.text(1.6, 3.9, "Stress tests", fontsize=6.0, weight="bold", va="center")
    ax.text(11.4, 3.9, "leave-one-client-out models  ·  PAD-UFES-20 smartphone images (domain shift)",
            fontsize=5.5, color=INK2, va="center")
    ax.plot([57.6, 57.6], [1.8, 6.0], color=AXIS, lw=0.6)
    ax.text(59.0, 3.9, "Fixed protocol", fontsize=6.0, weight="bold", va="center")
    ax.text(68.6, 3.9, "versions v0.1–v0.12 → tagged code → runs → pre-specified tests",
            fontsize=5.4, color=INK2, va="center")
    save(fig, "Fig2_framework")

# ---------------------------------------------------------------- Fig. S1 proximity examples (supplementary)
def figS_proximity(encoder="panderm", seed=0, n_leaked=4, n_new=2):
    """Randomly chosen image-level test images (seed 0 split) and their nearest training image in the frozen
    encoder's embedding space (centred, cosine). Needs embeddings/<encoder>.npz and the images; skipped otherwise.
    Selection is random (fixed seed), not the most similar pairs."""
    emb = os.path.join("embeddings", f"{encoder}.npz")
    if not os.path.exists(emb) or not os.path.isdir(IMG_DIR):
        print(f"  skip Fig. S1: needs {emb} and {IMG_DIR}")
        return
    z = np.load(emb)
    names, X = z["image"].astype(str), z["plain"].astype(np.float32)
    pos = {n: i for i, n in enumerate(names)}
    sp = pd.read_csv(os.path.join("splits", "e1", f"resplit_image_seed{seed}_val.csv"))
    tr, te = sp[sp.fold == "train"].reset_index(drop=True), sp[sp.fold == "test"].reset_index(drop=True)
    itr, ite = tr.image.astype(str).map(pos).to_numpy(), te.image.astype(str).map(pos).to_numpy()
    X = X - X[itr].mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    train_les = set(tr.lesion_id)
    rng = np.random.default_rng(20261106)
    leaked = np.flatnonzero(te.lesion_id.isin(train_les).to_numpy())
    new = np.flatnonzero(~te.lesion_id.isin(train_les).to_numpy())
    picks = list(rng.choice(leaked, n_leaked, replace=False)) + list(rng.choice(new, n_new, replace=False))
    rows = []
    for i in picks:
        sim = X[itr] @ X[ite[i]]
        j = int(sim.argmax())
        rows.append((te.image[i], tr.image[j], float(sim[j]), te.lesion_id[i] == tr.lesion_id[j], i in leaked))
    fig, ax = canvas(DOUBLE, 1.62)
    n = len(rows)
    w = 100 / n
    for k, (a, b, sv, same, lk) in enumerate(rows):
        x0 = k * w + 1.0
        thumb(ax, lesion_img(a), x0, 12.0, 7.0, ec=IMAGE if lk else LESION, lw=1.2, r=0.6)
        thumb(ax, lesion_img(b), x0 + 7.6, 12.0, 7.0, r=0.6)
        ax.text(x0 + 3.5, 20.0, "test", ha="center", fontsize=5.6, color=IMAGE if lk else LESION, weight="bold")
        ax.text(x0 + 11.1, 20.0, "nearest train", ha="center", fontsize=5.6, color=INK2)
        ax.text(x0 + 7.3, 10.6, f"cosine {sv:.2f}", ha="center", va="top", fontsize=5.8)
        ax.text(x0 + 7.3, 8.2, "same lesion" if same else "different lesion", ha="center", va="top", fontsize=5.6,
                color=IMAGE if same else INK2, weight="bold" if same else "normal")
        ax.text(x0 + 7.3, 5.9, "leaked test image" if lk else "new-lesion test image", ha="center", va="top",
                fontsize=5.2, color=MUTED)
    ax.plot([n_leaked * w, n_leaked * w], [4.5, 21.5], color=AXIS, lw=0.7, ls=(0, (3, 2)))
    save(fig, "FigS1_proximity_examples")


if __name__ == "__main__":
    fig1()
    fig_framework()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    figS_proximity()
