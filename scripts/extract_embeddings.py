"""
Frozen-encoder embeddings for model A-frozen (EfficientNet-B0, ImageNet) and model B (PanDerm-Base),
protocol v0.10, Section 4.

For every one of the 23,247 Fed-ISIC2019 images:
  * 1 plain view  (FLamby test transform: CenterCrop 200 + Normalize), used for validation and test;
  * 5 augmented views (FLamby training transforms), used for training the linear heads.
The augmented views are reproducible: view v of image i always uses the random seed 1,000,003 * v + i.
Both encoders see exactly the same pixels. PanDerm-Base (ViT-B/16) needs 224 x 224 input, so its
200 x 200 view is resized to 224 x 224 (bilinear); EfficientNet-B0 gets the 200 x 200 view as in model A.

Encoders (frozen, eval mode, fp16 autocast):
  effb0    EfficientNet-B0 (efficientnet-pytorch, ImageNet), global-average-pooled features, 1280-d
  panderm  PanDerm-Base, CLS token after the final LayerNorm (PanDerm's own linear-probe setting), 768-d

PanDerm code and weights are not part of this repository (CC-BY-NC-ND 4.0). Clone
https://github.com/SiyuanYan1/PanDerm and download PanDerm_Base (panderm_bb_data6_checkpoint-499.pth)
from the link in its README, then pass --panderm-code and --panderm-ckpt.

Output (fp16): embeddings/<encoder>.npz with image (N), plain (N x D), aug (N x 5 x D).

Usage (from the repo folder):
  python scripts\\extract_embeddings.py --smoke
  python scripts\\extract_embeddings.py
"""
import argparse
import importlib.util
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedderm.data import flamby_test_aug, flamby_train_aug  # noqa: E402
from fedderm.model import Baseline  # noqa: E402

N_AUG = 5


class ViewDataset(torch.utils.data.Dataset):
    """Returns the plain view and the 5 augmented views of one image (6 x 3 x 200 x 200)."""

    def __init__(self, images, img_dir):
        self.images = images
        self.img_dir = img_dir
        self.test_aug = flamby_test_aug()
        self.train_aug = None              # built in the worker (FLamby draws its CoarseDropout count once)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.train_aug is None:
            random.seed(20261031)          # same CoarseDropout setting in every worker and every run
            self.train_aug = flamby_train_aug()
        img = np.array(Image.open(os.path.join(self.img_dir, f"{self.images[i]}.jpg")))
        views = [self.test_aug(image=img)["image"]]
        for v in range(1, N_AUG + 1):
            s = 1_000_003 * v + i
            random.seed(s)
            np.random.seed(s % (2 ** 32))
            views.append(self.train_aug(image=img)["image"])
        x = np.stack([np.transpose(t, (2, 0, 1)) for t in views]).astype(np.float32)
        return torch.from_numpy(x), i


def load_effb0(device):
    m = Baseline(pretrained=True)
    m.base_model._fc = nn.Identity()
    return m.eval().to(device), 1280


def load_panderm(code_dir, ckpt, device):
    path = os.path.join(code_dir, "classification", "models", "modeling_finetune.py")
    spec = importlib.util.spec_from_file_location("panderm_modeling_finetune", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                  # timm's @register_model looks the module up here
    spec.loader.exec_module(mod)
    model = mod.panderm_base_patch16_224()
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {k[len("encoder."):] if k.startswith("encoder.") else k: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if not k.startswith("head.")]
    print(f"PanDerm: {len(sd)} tensors in checkpoint | missing (non-head) {len(missing)} | unexpected {len(res.unexpected_keys)}")
    if missing:
        raise RuntimeError(f"PanDerm weights do not match the model, e.g. missing {missing[:5]}")
    model.head = nn.Identity()
    return model.eval().to(device), 768


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join("splits", "split_official.csv"))
    ap.add_argument("--img-dir", default=os.path.join("data", "fedisic", "ISIC_2019_Training_Input_preprocessed"))
    ap.add_argument("--out", default="embeddings")
    ap.add_argument("--encoders", nargs="+", default=["effb0", "panderm"], choices=["effb0", "panderm"])
    ap.add_argument("--panderm-code", default=os.path.join("..", "ext", "PanDerm"))
    ap.add_argument("--panderm-ckpt", default=os.path.join("..", "models", "panderm", "panderm_bb_data6_checkpoint-499.pth"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    images = sorted(pd.read_csv(a.split).image.astype(str).unique())
    assert a.smoke or len(images) == 23247, f"expected 23,247 images, found {len(images)}"
    if a.smoke:
        images, a.workers = images[:64], 0
        a.out = a.out + "_smoke"
    os.makedirs(a.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoders = {}
    if "effb0" in a.encoders:
        encoders["effb0"] = load_effb0(device)
    if "panderm" in a.encoders:
        encoders["panderm"] = load_panderm(a.panderm_code, a.panderm_ckpt, device)
    for p in (q for m, _ in encoders.values() for q in m.parameters()):
        p.requires_grad_(False)

    n = len(images)
    store = {k: {"plain": np.zeros((n, d), np.float16), "aug": np.zeros((n, N_AUG, d), np.float16)}
             for k, (_, d) in encoders.items()}
    loader = torch.utils.data.DataLoader(ViewDataset(images, a.img_dir), batch_size=a.batch, shuffle=False,
                                         num_workers=a.workers, pin_memory=device.type == "cuda")
    t0, done = time.time(), 0
    with torch.no_grad():
        for x, idx in loader:
            b = x.shape[0]
            x = x.view(b * (N_AUG + 1), 3, 200, 200).to(device, non_blocking=True)
            for k, (m, d) in encoders.items():
                inp = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False) if k == "panderm" else x
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    e = m(inp).float().view(b, N_AUG + 1, d).cpu().numpy()
                store[k]["plain"][idx.numpy()] = e[:, 0]
                store[k]["aug"][idx.numpy()] = e[:, 1:]
            done += b
            if done % (a.batch * 50) < a.batch or done == n:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done}/{n} images ({rate:.0f} img/s, about {(n - done) / max(rate, 1e-6) / 60:.0f} min left)")

    for k, s in store.items():
        assert np.isfinite(s["plain"]).all() and np.isfinite(s["aug"]).all(), f"non-finite values in {k}"
        path = os.path.join(a.out, f"{k}.npz")
        np.savez(path, image=np.array(images), plain=s["plain"], aug=s["aug"])
        print(f"saved {path}: plain {s['plain'].shape}, aug {s['aug'].shape}, "
              f"mean |x| {np.abs(s['plain'].astype(np.float32)).mean():.3f}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
