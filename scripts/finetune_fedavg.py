"""
FedAvg-FT (protocol v0.10, Section 4: "Fine-tuning (FedAvg-FT): 3 local steps"), model A.

Starts from a finished FedAvg run (runs/<split>/fedavg_r9_s<seed>/): the global parameters (model.pt) plus
each client's own BatchNorm statistics (bn_buffers.pt), which is the state every client holds at the end
of FedAvg. Each client then makes 3 local SGD updates (lr 0.01, batch 64, FLamby augmentations, weighted
focal loss), as FLamby's FedAvgFineTuning does after its last round, and each center is evaluated with
its own fine-tuned model. (FLamby's benchmark evaluates only client 0's fine-tuned model on all
centers; per-center evaluation is the usual meaning of personalization and is what E2/E7 need.)

No new federated training is needed, so FedAvg-FT reuses every FedAvg run, including E1's.

Usage (from the repo folder):
  python scripts\\finetune_fedavg.py --split splits\\e1\\resplit_image_seed0_val.csv --seed 0
Output: runs/<split>/fedavgft_r9_s<seed>/ with metrics.json, preds.npz, preds_val.npz,
models_per_center.pt; a line in runs/runs_log.csv.
"""
import argparse
import csv
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedderm.data import NUM_CLIENTS, InfiniteLoader, IsicDataset, make_loader  # noqa: E402
from fedderm.evaluate import predict, summarize  # noqa: E402
from fedderm.fl import Client  # noqa: E402
from fedderm.model import Baseline, WeightedFocalLoss  # noqa: E402
from train_fl import git_commit, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--img-dir", default=os.path.join("data", "fedisic", "ISIC_2019_Training_Input_preprocessed"))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    split_name = os.path.splitext(os.path.basename(a.split))[0]
    src = os.path.join(a.out, split_name, f"fedavg_r{a.rounds}_s{a.seed}")
    out_dir = os.path.join(a.out, split_name, f"fedavgft_r{a.rounds}_s{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    glob_sd = torch.load(os.path.join(src, "model.pt"), map_location="cpu")
    bn = torch.load(os.path.join(src, "bn_buffers.pt"), map_location="cpu")
    t0 = time.time()

    models = []
    for c in range(NUM_CLIENTS):
        set_seed(a.seed + 1000 * c)
        m = Baseline(pretrained=False)
        sd = dict(glob_sd)
        sd.update(bn[c])                                    # this client's own BatchNorm statistics
        m.load_state_dict(sd)
        ds = IsicDataset(a.split, a.img_dir, "train", center=c, train_aug=True)
        loader = InfiniteLoader(make_loader(ds, True, a.batch, a.workers, a.seed + 1000 * c + 7))
        cl = Client(m, WeightedFocalLoss(), loader, 0.01, device, amp=True)
        cl.local_train(a.steps)
        models.append(cl.model.cpu())
        del loader

    per_center, rows, vrows, vlog, vy = {}, [], [], [], []
    for c in range(NUM_CLIENTS):
        mc = models[c].to(device)
        ds = IsicDataset(a.split, a.img_dir, "test", center=c, train_aug=False)
        lg, y = predict(mc, make_loader(ds, False, a.batch, a.workers, 0), device, True)
        per_center[c] = (lg, y)
        rows += [(img, c) for img in ds.df.image]
        dv = IsicDataset(a.split, a.img_dir, "val", center=c, train_aug=False)
        if len(dv):
            lv, yv = predict(mc, make_loader(dv, False, a.batch, a.workers, 0), device, True)
            vlog.append(lv)
            vy.append(yv)
            vrows += [(img, c) for img in dv.df.image]
        models[c] = mc.cpu()

    m = summarize(per_center)
    m.update({"method": "fedavgft", "seed": a.seed, "split": a.split, "rounds": a.rounds, "ft_steps": a.steps,
              "source_run": src, "eval_model": "own center's fine-tuned model", "git": git_commit(),
              "minutes": round((time.time() - t0) / 60, 1)})
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(out_dir, "preds.npz"), image=np.array([r[0] for r in rows]),
                        center=np.array([r[1] for r in rows]),
                        logits=np.concatenate([per_center[c][0] for c in range(NUM_CLIENTS)]),
                        target=np.concatenate([per_center[c][1] for c in range(NUM_CLIENTS)]))
    if vrows:
        np.savez_compressed(os.path.join(out_dir, "preds_val.npz"), image=np.array([r[0] for r in vrows]),
                            center=np.array([r[1] for r in vrows]), logits=np.concatenate(vlog),
                            target=np.concatenate(vy))
    torch.save([x.state_dict() for x in models], os.path.join(out_dir, "models_per_center.pt"))
    print("balanced accuracy  pooled {:.4f} | mean of centers {:.4f} | per center {}".format(
        m["bacc_pooled"], m["bacc_mean_of_centers"], " ".join(f"{m[f'bacc_center_{c}']:.3f}" for c in range(NUM_CLIENTS))))
    print(f"saved to {out_dir} ({m['minutes']} min)")
    with open(os.path.join(a.out, "runs_log.csv"), "a", newline="") as f:
        csv.writer(f).writerow([f"{dt.datetime.now():%Y-%m-%d %H:%M}", split_name, "fedavgft", a.seed, a.rounds,
                                f"{m['bacc_pooled']:.4f}", f"{m['bacc_mean_of_centers']:.4f}", m["minutes"],
                                m["git"], "ok"])


if __name__ == "__main__":
    main()
