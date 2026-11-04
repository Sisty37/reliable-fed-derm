"""
E3 MC-Dropout (protocol v0.11, Section 6, E3), model A FedAvg (9 rounds), both split types, seeds 0-4.

EfficientNet-B0 (efficientnet-pytorch) has one dropout layer, before the classifier (p = 0.2); its
drop-connect inside the blocks is tied to training mode together with BatchNorm, so only the final
dropout is sampled (BatchNorm stays in eval mode). The pooled features are computed once per image
(plain test view) and 20 dropout masks are applied to them, which is equivalent to 20 full forward
passes for this architecture. Saved per run: preds_mc.npz (test) and preds_mc_val.npz (validation) with
the mean of the 20 softmax outputs (T = 1) and the per-pass logits.

Usage (from the repo folder):  python scripts\\mc_dropout.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedderm.data import NUM_CLIENTS, IsicDataset, make_loader  # noqa: E402
from fedderm.model import Baseline  # noqa: E402

PASSES = 20


@torch.no_grad()
def features(model, loader, device):
    bm = model.base_model
    feats, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = bm.extract_features(x.to(device))
            f = bm._avg_pooling(f).flatten(1)
        feats.append(f.float())
        ys.append(y)
    return torch.cat(feats), torch.cat(ys).numpy()


def main():
    img_dir = os.path.join("data", "fedisic", "ISIC_2019_Training_Input_preprocessed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for kind in ("image", "lesion"):
        for s in range(5):
            split = os.path.join("splits", "e1", f"resplit_{kind}_seed{s}_val.csv")
            run = os.path.join("runs", f"resplit_{kind}_seed{s}_val", f"fedavg_r9_s{s}")
            if os.path.exists(os.path.join(run, "preds_mc.npz")):
                print(f"skip {run}")
                continue
            model = Baseline(pretrained=False)
            model.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location="cpu"))
            model = model.to(device).eval()
            p_drop = model.base_model._dropout.p
            gen = torch.Generator(device=device).manual_seed(1000 + s)
            for fold, fname in (("test", "preds_mc.npz"), ("val", "preds_mc_val.npz")):
                images, centers, F, Y = [], [], [], []
                for c in range(NUM_CLIENTS):
                    ds = IsicDataset(split, img_dir, fold, center=c, train_aug=False)
                    f, y = features(model, make_loader(ds, False, 64, 2, 0), device)
                    F.append(f)
                    Y.append(y)
                    images += list(ds.df.image)
                    centers += [c] * len(ds)
                F = torch.cat(F)
                logits = []
                for _ in range(PASSES):
                    mask = (torch.rand(F.shape, generator=gen, device=device) >= p_drop).float() / (1 - p_drop)
                    with torch.no_grad():
                        logits.append(model.base_model._fc(F * mask).detach().float().cpu().numpy())
                logits = np.stack(logits, 1)                                   # images x passes x classes
                e = np.exp(logits - logits.max(2, keepdims=True))
                probs = (e / e.sum(2, keepdims=True)).mean(1)
                np.savez_compressed(os.path.join(run, fname), image=np.array(images), center=np.array(centers),
                                    target=np.concatenate(Y), probs=probs, logits_passes=logits.astype(np.float16))
            print(f"saved MC-Dropout predictions in {run} (p = {p_drop}, {PASSES} passes)")


if __name__ == "__main__":
    main()

