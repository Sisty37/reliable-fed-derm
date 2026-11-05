"""
E6 PAD-UFES-20 stress test, predictions (protocol v0.11). No training: the lesion-grouped FedAvg models
of E5 (seeds 0-4) are applied unchanged to all 2,298 PAD-UFES-20 images.

  model A  EfficientNet-B0 FedAvg (runs/resplit_lesion_seed<s>_val/fedavg_r9_s<s>/model.pt), FLamby test
           transform (CenterCrop 200 + Normalize), eval mode (client 0's BatchNorm statistics, the same
           model that was evaluated on Fed-ISIC2019).
  model B  PanDerm-Base CLS embedding of the same 200 x 200 view resized to 224 (as in
           extract_embeddings.py), centred with the federated training mean of the same Fed-ISIC2019
           split (as in train_linear.py), then the FedAvg linear head
           (runs/resplit_lesion_seed<s>_val/pandermfrozen_fedavg_r9_s<s>/model.pt).

Outputs: embeddings/panderm_padufes20.npz (plain view, fp16) and
runs/padufes20/<run>/preds.npz (image, target, patient, lesion, logits over the 8 Fed-ISIC classes).

Usage (from the repo folder):  python scripts\\padufes_predict.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fedderm.data import IsicDataset, make_loader  # noqa: E402
from fedderm.model import Baseline  # noqa: E402
from extract_embeddings import load_panderm  # noqa: E402

SPLIT = os.path.join("splits", "padufes20.csv")
IMG_DIR = os.path.join("data", "padufes20", "preprocessed")


def isic_train_mean(seed, emb):
    """Federated training mean of the Fed-ISIC2019 embeddings (all 5 views), as train_linear.py."""
    sp = pd.read_csv(os.path.join("splits", "e1", f"resplit_lesion_seed{seed}_val.csv"))
    pos = {im: i for i, im in enumerate(emb["image"].astype(str))}
    rows = sp.loc[sp.fold == "train", "image"].astype(str).map(pos).to_numpy()
    x = emb["aug"][rows].astype(np.float64).reshape(-1, emb["aug"].shape[2])
    return x.mean(0).astype(np.float32)


def save(run, df, logits):
    d = os.path.join("runs", "padufes20", run)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "preds.npz"), image=df.image.to_numpy().astype(str),
                        target=df.target.to_numpy(), patient=df.patient_id.to_numpy().astype(str),
                        lesion=df.lesion_id.to_numpy(), logits=logits.astype(np.float32))
    print(f"saved runs/padufes20/{run}/preds.npz ({len(df)} images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--panderm-code", default=os.path.join("..", "ext", "PanDerm"))
    ap.add_argument("--panderm-ckpt", default=os.path.join("..", "models", "panderm", "panderm_bb_data6_checkpoint-499.pth"))
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = IsicDataset(SPLIT, IMG_DIR, "test")
    df = ds.df
    assert len(df) == 2298

    # model A
    for s in a.seeds:
        model = Baseline(pretrained=False)
        model.load_state_dict(torch.load(os.path.join("runs", f"resplit_lesion_seed{s}_val", f"fedavg_r9_s{s}", "model.pt"),
                                         map_location="cpu"))
        model = model.to(device).eval()
        out = []
        with torch.no_grad():
            for x, _ in make_loader(ds, False, 64, 2, 0):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    out.append(model(x.to(device)).float().cpu().numpy())
        save(f"fedavg_r9_s{s}", df, np.concatenate(out))

    # model B: PanDerm embeddings of the plain view (computed once)
    emb_path = os.path.join("embeddings", "panderm_padufes20.npz")
    if not os.path.exists(emb_path):
        enc, dim = load_panderm(a.panderm_code, a.panderm_ckpt, device)
        feats = []
        with torch.no_grad():
            for x, _ in make_loader(ds, False, 32, 2, 0):
                x = F.interpolate(x.to(device), size=(224, 224), mode="bilinear", align_corners=False)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    feats.append(enc(x).float().cpu().numpy())
        plain = np.concatenate(feats).astype(np.float16)
        assert np.isfinite(plain).all()
        np.savez(emb_path, image=df.image.to_numpy().astype(str), plain=plain)
        print(f"saved {emb_path}: {plain.shape}")
        del enc
    pad = np.load(emb_path)["plain"].astype(np.float32)
    z = np.load(os.path.join("embeddings", "panderm.npz"))
    isic = {"image": z["image"], "aug": z["aug"]}          # read once (npz members are re-read on every access)
    for s in a.seeds:
        mu = isic_train_mean(s, isic)
        head = nn.Linear(pad.shape[1], 8)
        head.load_state_dict(torch.load(os.path.join("runs", f"resplit_lesion_seed{s}_val", f"pandermfrozen_fedavg_r9_s{s}",
                                                     "model.pt"), map_location="cpu"))
        with torch.no_grad():
            logits = head(torch.from_numpy(pad - mu)).numpy()
        save(f"pandermfrozen_fedavg_r9_s{s}", df, logits)


if __name__ == "__main__":
    main()
