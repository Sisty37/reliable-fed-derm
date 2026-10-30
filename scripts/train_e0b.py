"""
E0b (protocol v0.10): train model A with FedAvg on one mother-paper pipeline (arm a, b or c) and one seed.

Training is the same as E0 (EfficientNet-B0, FLamby augmentations, FedAvg, 9 rounds x 100 local updates,
SGD lr 0.01, batch 64, final model = client 0 copy), with 3 Dirichlet clients from make_e0b_splits.py
and the 7 HAM10000 classes. Loss: focal loss (gamma 2) **without** class weights, because the training
set is already balanced by the pipeline (FLamby's class weights would re-weight a balanced set).
Copies are rebuilt from their source image with their flip / rotation (`tf`) before augmentation.

Reported on the arm's own test set: balanced accuracy (the arm's headline number), the leakage shares,
and, as an exploratory number, balanced accuracy on the test rows whose lesion is not in training.

Usage (from the repo folder):
  python scripts\\train_e0b.py --arm a --seed 0 --smoke     # 1-minute check
  python scripts\\train_e0b.py --arm a --seed 0
Outputs: runs/e0b/arm<a|b|c>_s<seed>/ (metrics.json, preds.npz, model.pt, train_log.txt) and a line in
runs/e0b/e0b_log.csv.
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedderm.data import InfiniteLoader, flamby_test_aug, flamby_train_aug, make_loader  # noqa: E402
from fedderm.evaluate import predict  # noqa: E402
from fedderm.fl import Federation  # noqa: E402
from fedderm.model import Baseline, WeightedFocalLoss  # noqa: E402
from train_fl import git_commit, set_seed  # noqa: E402

N_CLASSES = 7
N_CLIENTS = 3
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def dihedral(img, tf):
    """0 original, 1 h-flip, 2 v-flip, 3 rot90, 4 rot180, 5 rot270 (numpy HxWxC)."""
    if tf == 1:
        img = img[:, ::-1]
    elif tf == 2:
        img = img[::-1]
    elif tf in (3, 4, 5):
        img = np.rot90(img, k=tf - 2)
    return np.ascontiguousarray(img)


class E0bDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, train_aug):
        self.df = df.reset_index(drop=True)
        self.paths = [os.path.join(img_dir, f"{n}.jpg") for n in self.df.image]
        self.tfs = self.df.tf.to_numpy()
        self.targets = self.df.target.to_numpy()
        self.aug = flamby_train_aug() if train_aug else flamby_test_aug()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = dihedral(np.array(Image.open(self.paths[i])), int(self.tfs[i]))
        img = self.aug(image=img)["image"]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(img), torch.tensor(int(self.targets[i]), dtype=torch.int64)


def bacc(y, pred):
    return float(balanced_accuracy_score(y, pred)) if len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["a", "b", "c"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-dir", default=os.path.join("splits", "e0b"))
    ap.add_argument("--img-dir", default=os.path.join("data", "fedisic", "ISIC_2019_Training_Input_preprocessed"))
    ap.add_argument("--out", default=os.path.join("runs", "e0b"))
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.rounds, a.updates, a.workers = 2, 2, 0
        a.out = a.out + "_smoke"

    amp = not a.no_amp
    tag = f"arm{a.arm}_s{a.seed}"
    out_dir = os.path.join(a.out, tag)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train_log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{dt.datetime.now():%Y-%m-%d %H:%M} | E0b {tag} | device {device} | AMP {amp} | lr {a.lr} | git {git_commit()}")
    set_seed(a.seed)

    df = pd.read_csv(os.path.join(a.split_dir, f"arm{a.arm}_seed{a.seed}.csv"))
    train, test = df[df.fold == "train"], df[df.fold == "test"]
    if a.smoke:
        train = train.groupby("client", group_keys=False).head(40)
        test = test.head(200)

    loaders = []
    for c in range(N_CLIENTS):
        ds = E0bDataset(train[train.client == c], a.img_dir, train_aug=True)
        loaders.append(InfiniteLoader(make_loader(ds, True, a.batch, a.workers, a.seed + 1000 * c)))
    log("train rows per client: " + ", ".join(str(l.n) for l in loaders) + f" | test rows: {len(test)}")

    model = Baseline(pretrained=True, num_classes=N_CLASSES)
    loss_fn = WeightedFocalLoss(alpha=torch.ones(N_CLASSES))     # unweighted focal, gamma 2
    fed = Federation(model, loss_fn, loaders, a.lr, device, amp, "fedavg")
    t0 = time.time()
    for r in range(a.rounds):
        losses = fed.round(a.updates)
        log(f"  round {r + 1}/{a.rounds}  client losses " + " ".join(f"{x:.3f}" for x in losses)
            + f"  ({time.time() - t0:.0f} s)")
    eval_model = fed.clients[0].model

    test_ds = E0bDataset(test, a.img_dir, train_aug=False)
    logits, y = predict(eval_model, make_loader(test_ds, False, a.batch, a.workers, 0), device, amp)
    pred = logits.argmax(1)
    tr_lesions, tr_images = set(train.lesion_id), set(train.image)
    lesion_seen = test.lesion_id.isin(tr_lesions).to_numpy()
    source_seen = test.image.isin(tr_images).to_numpy()

    m = {
        "arm": a.arm, "seed": a.seed, "rounds": a.rounds, "updates": a.updates, "lr": a.lr, "amp": amp,
        "git": git_commit(), "test_rows": int(len(y)),
        "bacc_test": bacc(y, pred),
        "share_test_lesion_in_train": float(lesion_seen.mean()),
        "share_test_source_image_in_train": float(source_seen.mean()),
        "bacc_test_new_lesions_exploratory": bacc(y[~lesion_seen], pred[~lesion_seen]),
        "n_test_new_lesions": int((~lesion_seen).sum()),
        "recall_per_class": dict(zip(CLASS_NAMES, recall_score(y, pred, labels=range(N_CLASSES),
                                                                average=None, zero_division=0).round(4).tolist())),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), image=test.image.to_numpy().astype(str),
                        tf=test.tf.to_numpy(), lesion_id=test.lesion_id.to_numpy().astype(str),
                        logits=logits, target=y)
    torch.save(eval_model.state_dict(), os.path.join(out_dir, "model.pt"))

    log(f"balanced accuracy (own test set) {m['bacc_test']:.4f} | test rows with lesion in train "
        f"{m['share_test_lesion_in_train']:.1%} | with source image in train {m['share_test_source_image_in_train']:.1%}")
    log(f"exploratory: balanced accuracy on {m['n_test_new_lesions']} new-lesion test rows "
        f"{m['bacc_test_new_lesions_exploratory']:.4f}")
    log(f"saved to {out_dir} ({m['minutes']} min)")

    log_path = os.path.join(a.out, "e0b_log.csv")
    new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "arm", "seed", "bacc_test", "share_lesion_in_train", "share_source_in_train",
                        "bacc_new_lesions", "minutes", "git"])
        w.writerow([f"{dt.datetime.now():%Y-%m-%d %H:%M}", a.arm, a.seed, f"{m['bacc_test']:.4f}",
                    f"{m['share_test_lesion_in_train']:.4f}", f"{m['share_test_source_image_in_train']:.4f}",
                    f"{m['bacc_test_new_lesions_exploratory']:.4f}", m["minutes"], m["git"]])


if __name__ == "__main__":
    main()
