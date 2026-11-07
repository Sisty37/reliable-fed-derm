"""
Contact sheet of candidate lesions for Fig. 1a: melanomas of client 0 with exactly three images, of which
two are training and one is a test image in the official FLamby split (a real leaked lesion).

Usage (from the repo folder):  python scripts\\fig1_candidates.py
Writes figures/Fig1_candidates.png; pick a lesion and run
  $env:FIG_LESION="BCN_xxxxxxx"; python scripts\\make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

IMG_DIR = os.path.join("data", "fedisic", "ISIC_2019_Training_Input_preprocessed")
sp = pd.read_csv(os.path.join("splits", "split_official.csv"))
g = sp[(sp.center == 0) & (sp.target == 0)].groupby("lesion_id")
cands = [(l, x) for l, x in g if len(x) == 3 and (x.fold == "test").sum() == 1]
fig, axes = plt.subplots(len(cands), 3, figsize=(4.5, 1.6 * len(cands)))
for r, (les, x) in enumerate(cands):
    imgs = x[x.fold == "train"].image.tolist() + x[x.fold == "test"].image.tolist()
    for c, name in enumerate(imgs):
        ax = axes[r, c]
        ax.imshow(Image.open(os.path.join(IMG_DIR, f"{name}.jpg")))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{c + 1} ({'test' if c == 2 else 'train'})", fontsize=6)
    axes[r, 0].set_ylabel(les, fontsize=7, rotation=0, ha="right", va="center")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig(os.path.join("figures", "Fig1_candidates.png"), dpi=110)
print(f"saved figures/Fig1_candidates.png ({len(cands)} lesions)")
