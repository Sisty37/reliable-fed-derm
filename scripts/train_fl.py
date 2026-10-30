"""
Train one run (one method, one seed, one split) and save metrics, test predictions and the model.

Examples (from the repo folder, Windows PowerShell):
  # 2-3 minute pipeline check on a few images
  python scripts\\train_fl.py --method fedavg --seed 0 --smoke
  # E0: official split
  python scripts\\train_fl.py --method fedavg   --seed 0
  python scripts\\train_fl.py --method fedprox  --seed 0
  python scripts\\train_fl.py --method scaffold --seed 0
  python scripts\\train_fl.py --method pooled   --seed 0

Outputs: runs/<split>/<method>_r<rounds>_s<seed>/ with metrics.json, preds.npz, model.pt,
bn_buffers.pt (federated runs), train_log.txt; one line appended to runs/runs_log.csv.
"""
import argparse
import csv
import datetime as dt
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedderm.data import NUM_CLIENTS, InfiniteLoader, IsicDataset, make_loader  # noqa: E402
from fedderm.evaluate import predict, summarize  # noqa: E402
from fedderm.fl import Federation, train_pooled  # noqa: E402
from fedderm.model import Baseline, WeightedFocalLoss  # noqa: E402


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fedavg", "fedprox", "scaffold", "pooled"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=os.path.join("splits", "split_official.csv"))
    ap.add_argument("--img-dir", default=os.path.join("data", "fedisic", "ISIC_2019_Training_Input_preprocessed"))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--rounds", type=int, default=9)          # FLamby get_nb_max_rounds(100)
    ap.add_argument("--updates", type=int, default=100)       # local updates per round
    ap.add_argument("--lr", type=float, default=None)         # 0.01 (SGD) for FL, 5e-4 (Adam) pooled
    ap.add_argument("--mu", type=float, default=0.001)        # FedProx
    ap.add_argument("--server-lr", type=float, default=1.0)   # SCAFFOLD
    ap.add_argument("--epochs", type=int, default=20)         # pooled
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to test the pipeline")
    a = ap.parse_args()

    if a.smoke:
        a.rounds, a.updates, a.epochs, a.workers = 2, 2, 1, 0
    amp = not a.no_amp
    lr = a.lr if a.lr is not None else (5e-4 if a.method == "pooled" else 0.01)
    split_name = os.path.splitext(os.path.basename(a.split))[0] + ("_smoke" if a.smoke else "")
    tag = f"{a.method}_r{a.rounds}_s{a.seed}" if a.method != "pooled" else f"pooled_e{a.epochs}_s{a.seed}"
    out_dir = os.path.join(a.out, split_name, tag)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train_log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{dt.datetime.now():%Y-%m-%d %H:%M} | {tag} | split {a.split} | device {device}"
        f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''} | AMP {amp} | lr {lr}"
        f" | git {git_commit()}")
    set_seed(a.seed)
    max_items = 40 if a.smoke else None

    model = Baseline(pretrained=True)
    loss_fn = WeightedFocalLoss()
    t0 = time.time()

    if a.method == "pooled":
        ds = IsicDataset(a.split, a.img_dir, "train", train_aug=True, max_items=max_items)
        log(f"pooled train images: {len(ds)}")
        model = train_pooled(model, loss_fn, make_loader(ds, True, a.batch, a.workers, a.seed),
                             a.epochs, lr, device, amp, log)
        eval_model, bn = model, None
    else:
        loaders = []
        for c in range(NUM_CLIENTS):
            ds = IsicDataset(a.split, a.img_dir, "train", center=c, train_aug=True, max_items=max_items)
            loaders.append(InfiniteLoader(make_loader(ds, True, a.batch, a.workers, a.seed + 1000 * c)))
        log("train images per center: " + ", ".join(str(l.n) for l in loaders))
        fed = Federation(model, loss_fn, loaders, lr, device, amp, a.method, mu=a.mu, server_lr=a.server_lr)
        for r in range(a.rounds):
            losses = fed.round(a.updates)
            log(f"  round {r + 1}/{a.rounds}  client losses " + " ".join(f"{x:.3f}" for x in losses)
                + f"  ({time.time() - t0:.0f} s)")
        eval_model = fed.clients[0].model                  # FLamby convention
        bn = [{k: v.cpu() for k, v in c.model.state_dict().items() if "running" in k or "num_batches" in k}
              for c in fed.clients]

    per_center, rows = {}, []
    for c in range(NUM_CLIENTS):
        ds = IsicDataset(a.split, a.img_dir, "test", center=c, train_aug=False, max_items=max_items)
        lg, y = predict(eval_model, make_loader(ds, False, a.batch, a.workers, 0), device, amp)
        per_center[c] = (lg, y)
        rows += [(img, c) for img in ds.df.image]
    metrics = summarize(per_center)
    metrics.update({"method": a.method, "seed": a.seed, "split": a.split, "rounds": a.rounds,
                    "updates": a.updates, "lr": lr, "amp": amp, "minutes": round((time.time() - t0) / 60, 1),
                    "git": git_commit(), "eval_model": "pooled" if a.method == "pooled" else "client 0 copy"})
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(out_dir, "preds.npz"),
                        image=np.array([r[0] for r in rows]), center=np.array([r[1] for r in rows]),
                        logits=np.concatenate([per_center[c][0] for c in range(NUM_CLIENTS)]),
                        target=np.concatenate([per_center[c][1] for c in range(NUM_CLIENTS)]))
    torch.save(eval_model.state_dict(), os.path.join(out_dir, "model.pt"))
    if bn is not None:
        torch.save(bn, os.path.join(out_dir, "bn_buffers.pt"))

    log("balanced accuracy  pooled {:.4f} | mean of centers {:.4f} | per center {}".format(
        metrics["bacc_pooled"], metrics["bacc_mean_of_centers"],
        " ".join(f"{metrics[f'bacc_center_{c}']:.3f}" for c in range(NUM_CLIENTS))))
    log(f"saved to {out_dir} ({metrics['minutes']} min)")

    log_path = os.path.join(a.out, "runs_log.csv")
    new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "split", "method", "seed", "rounds", "bacc_pooled", "bacc_mean_centers", "minutes", "git", "status"])
        w.writerow([f"{dt.datetime.now():%Y-%m-%d %H:%M}", split_name, a.method, a.seed, a.rounds,
                    f"{metrics['bacc_pooled']:.4f}", f"{metrics['bacc_mean_of_centers']:.4f}", metrics["minutes"],
                    metrics["git"], "smoke" if a.smoke else "ok"])


if __name__ == "__main__":
    main()
