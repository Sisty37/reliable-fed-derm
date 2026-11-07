"""
Federated / pooled linear heads on frozen embeddings: model A-frozen (--encoder effb0) and model B
(--encoder panderm), protocol v0.10, Section 4.

  * Embeddings are centred per dimension with the federated training mean (default; deviation of
    31 Oct 2026, see results/E1_frozen_implementation_notes.md). --feature-norm zscore / none keep the
    earlier variants for the record.
  * Training rows use one of their 5 augmented-view embeddings, drawn at random each time a row is used;
    validation and test use the plain view.
  * Loss: FLamby weighted focal loss (gamma 2). FedAvg: SGD lr 0.01, batch 64, 9 rounds x U local
    updates per client, final model = client 0 copy (as model A). Pooled: Adam lr 5e-4, batch 64,
    20 epochs.
  * U (local updates per round) comes from the convergence check (--convergence-check), run once per
    encoder on the lesion-grouped split, seed 0, FedAvg: starting at 100, U is doubled while the pooled
    training loss falls by more than 1% over the last round. The chosen U is saved to
    results/linear_updates_<encoder>.json and used unchanged for every later run of that encoder.

  * FedPer-style head (E7, model B): Linear(dim, 256) - ReLU - Linear(256, 8); the hidden layer is
    federated (averaged as in FedAvg), the output layer is client-specific; each center is evaluated
    with its own client's head. Same optimizer, rounds and U as the FedAvg linear head (U is not
    re-tuned; choice fixed in results/E7_implementation_notes.md).

Outputs as train_fl.py: runs/<split>/<encoder>frozen_<method>_r9_s<seed>/ with metrics.json, preds.npz,
preds_val.npz, model.pt; a line in runs/runs_log.csv.

Usage (from the repo folder):
  python scripts\\train_linear.py --encoder effb0 --convergence-check
  python scripts\\train_linear.py --encoder effb0 --method fedavg --seed 0 --split splits\\e1\\resplit_image_seed0_val.csv
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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fedderm.evaluate import summarize  # noqa: E402
from fedderm.fl import Client, Federation, personal_check  # noqa: E402
from fedderm.model import WeightedFocalLoss  # noqa: E402
from train_fl import git_commit, set_seed  # noqa: E402

NUM_CLIENTS, NUM_CLASSES, N_AUG = 6, 8, 5


class EmbRows:
    """Embeddings of one set of rows; .sample(n) draws a random batch with a random augmented view."""

    def __init__(self, aug, y, rng):
        self.aug, self.y, self.rng = aug, y, rng
        self.n = len(y)
        self.order, self.pos = rng.permutation(self.n), 0

    def next(self, batch=64):                     # cycles through the rows like FLamby's loader
        if self.pos >= self.n:
            self.order, self.pos = self.rng.permutation(self.n), 0
        idx = self.order[self.pos:self.pos + batch]
        self.pos += batch
        v = self.rng.integers(0, N_AUG, size=len(idx))
        return torch.from_numpy(self.aug[idx, v].astype(np.float32)), torch.from_numpy(self.y[idx])


def federated_standardize(df, plain, aug, eps=1e-4, mode="center"):
    """Per-dimension z-scoring with training statistics aggregated federatedly: each client shares only
    its row count, sum and sum of squares over its training rows (all 5 views); the server combines
    them. The same transform is applied to every row and view (train, val, test)."""
    n = 0
    s1 = np.zeros(plain.shape[1], np.float64)
    s2 = np.zeros(plain.shape[1], np.float64)
    tr = df[df.fold == "train"]
    for c in sorted(tr.center.unique()):                      # client-side statistics
        x = aug[tr.row[tr.center == c].to_numpy()].astype(np.float64).reshape(-1, plain.shape[1])
        n += len(x)
        s1 += x.sum(0)
        s2 += (x ** 2).sum(0)
    mu = s1 / n                                               # server-side combination
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0) + eps)
    mu, sd = mu.astype(np.float32), sd.astype(np.float32)
    if mode == "none":
        return plain.astype(np.float32), aug.astype(np.float32)
    if mode == "center":                                      # remove offsets, keep the scale
        return plain.astype(np.float32) - mu, aug.astype(np.float32) - mu
    return (plain.astype(np.float32) - mu) / sd, (aug.astype(np.float32) - mu) / sd


def load_split(split_csv, emb_path):
    e = np.load(emb_path)
    pos = {im: i for i, im in enumerate(e["image"].astype(str))}
    df = pd.read_csv(split_csv)
    df["row"] = df.image.astype(str).map(pos)
    assert df.row.notna().all(), "split images missing from the embedding file"
    return df, e["plain"], e["aug"]


@torch.no_grad()
def predict(model, x, device):
    model.eval()
    return model(torch.from_numpy(x.astype(np.float32)).to(device)).cpu().numpy()


@torch.no_grad()
def train_loss(model, loss_fn, aug, y, device):
    """Pooled training loss over all training rows and all 5 views (convergence check)."""
    model.eval()
    tot = 0.0
    for v in range(N_AUG):
        x = torch.from_numpy(aug[:, v].astype(np.float32)).to(device)
        tot += loss_fn(model(x), torch.from_numpy(y).to(device)).item()
    return tot / N_AUG


def run(a, updates, log):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    df, plain, aug = load_split(a.split, os.path.join(a.emb_dir, f"{a.encoder}.npz"))
    df["row"] = df.row.astype(int)
    if a.exclude_center is not None:                 # leave-one-center-out: drop that center's training rows
        df.loc[(df.center == a.exclude_center) & (df.fold == "train"), "fold"] = "excluded"
    plain, aug = federated_standardize(df, plain, aug, mode=a.feature_norm)
    tr = df[df.fold == "train"]
    dim = plain.shape[1]

    def make_head():                             # --head mlp: capacity-matched to the FedPer-style head (E9)
        if a.head == "mlp":
            return nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, NUM_CLASSES))
        return nn.Linear(dim, NUM_CLASSES)       # PyTorch default initialization (not zero)
    torch.manual_seed(a.seed)
    head = make_head()
    loss_fn = WeightedFocalLoss()
    tr_aug, tr_y = aug[tr.row.to_numpy()], tr.target.to_numpy().astype(np.int64)
    losses = []
    t0 = time.time()
    def client_rows(c, seed_off=0):
        g = tr[tr.center == c]
        return EmbRows(aug[g.row.to_numpy()], g.target.to_numpy().astype(np.int64),
                       np.random.default_rng(a.seed + 1000 * c + seed_off))

    def adam_epochs(m, rows, n_rows):
        m, lf = m.to(device), loss_fn.to(device)
        opt = torch.optim.Adam(m.parameters(), lr=5e-4)
        for ep in range(a.epochs):
            m.train()
            for _ in range(int(np.ceil(n_rows / 64))):
                x, y = rows.next()
                loss = lf(m(x.to(device)), y.to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return m

    if a.method == "pooled":
        head = adam_epochs(head, EmbRows(tr_aug, tr_y, rng), len(tr))
        losses.append(train_loss(head, loss_fn.to(device), tr_aug, tr_y, device))
        log(f"  pooled: {a.epochs} epochs, final training loss {losses[-1]:.4f}")
        models = [head] * NUM_CLIENTS
    elif a.method == "local":                    # local-only: pooled recipe on one center's rows
        models = []
        for c in range(NUM_CLIENTS):
            torch.manual_seed(a.seed + 1000 * c)
            n_c = int((tr.center == c).sum())
            models.append(adam_epochs(make_head(), client_rows(c), n_c))
        log(f"  local-only: {a.epochs} epochs per center")
    elif a.method == "fedavgft":                 # 3 local SGD steps from the finished FedAvg head
        split_name = os.path.splitext(os.path.basename(a.split))[0]
        src = os.path.join(a.out, split_name, f"{a.encoder}frozen_fedavg_r{a.rounds}_s{a.seed}", "model.pt")
        head.load_state_dict(torch.load(src, map_location="cpu"))
        models = []
        for c in range(NUM_CLIENTS):
            torch.manual_seed(a.seed + 1000 * c)
            cl = Client(head, loss_fn, client_rows(c, seed_off=7), 0.01, device, amp=False)
            cl.local_train(a.ft_steps)
            models.append(cl.model)
        log(f"  FedAvg-FT: {a.ft_steps} local steps per client from {src}")
    elif a.method == "fedper":                   # FedPer-style two-layer head, client-specific output layer
        torch.manual_seed(a.seed)
        mlp = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, NUM_CLASSES))
        loaders = [client_rows(c) for c in range(NUM_CLIENTS)]
        fed = Federation(mlp, loss_fn, loaders, 0.01, device, amp=False, strategy="fedper",
                         local_names=["2.weight", "2.bias"])
        for r in range(a.rounds):
            fed.round(updates)
            losses.append(float(np.mean([c.last_loss for c in fed.clients])))
        log(f"  fedper: {a.rounds} rounds x {updates} updates, mean client training loss by round "
            + " ".join(f"{x:.4f}" for x in losses))
        sh, lo, ln = personal_check(fed)
        log(f"  check: max |client difference| shared {sh:.2e} (must be 0), local {lo:.2e} (must be > 0); local {ln}")
        models = [c.model for c in fed.clients]
    else:                                        # fedavg, fedprox, scaffold
        loaders = [client_rows(c) for c in range(NUM_CLIENTS) if c != a.exclude_center]
        fed = Federation(head, loss_fn, loaders, 0.01, device, amp=False, strategy=a.method,
                         mu=0.001, server_lr=1.0)
        for r in range(a.rounds):
            fed.round(updates)
            losses.append(train_loss(fed.clients[0].model, fed.clients[0].loss_fn, tr_aug, tr_y, device))
        log(f"  {a.method}: {a.rounds} rounds x {updates} updates, pooled training loss by round "
            + " ".join(f"{x:.4f}" for x in losses))
        models = [fed.clients[0].model] * NUM_CLIENTS
    return models, df, plain, losses, device, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["effb0", "panderm"])
    ap.add_argument("--method", default="fedavg", choices=["fedavg", "fedprox", "scaffold", "fedavgft", "fedper", "pooled", "local"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=os.path.join("splits", "e1", "resplit_lesion_seed0_val.csv"))
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--convergence-check", action="store_true")
    ap.add_argument("--ft-steps", type=int, default=3,
                    help="FedAvg-FT local steps (protocol: 3; other values = post hoc sensitivity, tag fedavgft<steps>)")
    ap.add_argument("--epochs", type=int, default=20,
                    help="pooled/local epochs (protocol: 20; other values = post hoc matched-budget control, tag _e<epochs>)")
    ap.add_argument("--exclude-center", type=int, default=None,
                    help="E6 leave-one-center-out: this center's training rows are not used (FedAvg only)")
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"],
                    help="E9: 'mlp' = Linear(dim,256)-ReLU-Linear(256,8) for fedavg/fedprox/scaffold/pooled/local, "
                         "capacity-matched to the FedPer-style head")
    ap.add_argument("--feature-norm", default="center", choices=["center", "zscore", "none"],
                    help="federated per-dimension centering (default), z-scoring, or raw embeddings")
    a = ap.parse_args()

    upd_file = os.path.join("results", f"linear_updates_{a.encoder}.json")
    if a.convergence_check:
        assert a.method == "fedavg" and a.seed == 0 and "lesion_seed0" in a.split, \
            "the convergence check is defined on the lesion-grouped split, seed 0, FedAvg"
        updates, history = 100, []
        while True:
            _, _, _, losses, _, _ = run(a, updates, print)
            drop = (losses[-2] - losses[-1]) / losses[-2]
            history.append({"updates": updates, "loss_last_two_rounds": losses[-2:], "relative_drop": drop})
            print(f"U = {updates}: loss fell {100 * drop:.2f}% over the last round")
            if drop <= 0.01 or updates >= 102400:
                break
            updates *= 2
        os.makedirs("results", exist_ok=True)
        json.dump({"encoder": a.encoder, "feature_norm": a.feature_norm, "updates": updates, "history": history, "git": git_commit(),
                   "date": f"{dt.datetime.now():%Y-%m-%d %H:%M}"}, open(upd_file, "w"), indent=2)
        print(f"chosen local updates per round for {a.encoder}: {updates} -> {upd_file}")
        return

    if a.method in ("fedavg", "fedprox", "scaffold", "fedper"):
        assert os.path.exists(upd_file), f"run the convergence check first ({upd_file} missing)"
        cfg = json.load(open(upd_file))
        assert cfg.get("feature_norm", "zscore") == a.feature_norm, f"{upd_file} was made with another --feature-norm"
        updates = cfg["updates"]
    else:
        updates = 0
    split_name = os.path.splitext(os.path.basename(a.split))[0]
    assert not (a.head == "mlp" and a.method in ("fedper", "fedavgft")), "fedper already has the two-layer head; fedavgft is linear only"
    mname = a.method + ("mlp" if a.head == "mlp" else "") + (str(a.ft_steps) if a.method == "fedavgft" and a.ft_steps != 3 else "")
    tag = f"{a.encoder}frozen_" + (f"{mname}_e{a.epochs}_s{a.seed}" if a.method in ("pooled", "local")
                                   else f"{mname}_r{a.rounds}_s{a.seed}")
    if a.exclude_center is not None:
        assert a.method == "fedavg", "leave-one-center-out is defined for FedAvg"
        tag = f"{a.encoder}frozen_{a.method}_r{a.rounds}_loco{a.exclude_center}_s{a.seed}"
    out_dir = os.path.join(a.out, split_name, tag)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "train_log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg)
        logf.write(msg + "\n")
        logf.flush()

    log(f"{dt.datetime.now():%Y-%m-%d %H:%M} | {tag} | split {a.split} | updates {updates} | git {git_commit()}")
    models, df, plain, losses, device, secs = run(a, updates, log)

    per_center = {}
    for fold, fname in (("test", "preds.npz"), ("val", "preds_val.npz")):
        part = df[df.fold == fold]
        if part.empty:
            continue
        cen = part.center.to_numpy()
        lg = np.zeros((len(part), NUM_CLASSES), np.float32)
        for c in range(NUM_CLIENTS):                      # each center with its own model
            if (cen == c).any():
                lg[cen == c] = predict(models[c], plain[part.row.to_numpy()[cen == c]], device)
        y = part.target.to_numpy()
        np.savez_compressed(os.path.join(out_dir, fname), image=part.image.to_numpy().astype(str),
                            center=part.center.to_numpy(), logits=lg, target=y)
        if fold == "test":
            per_center = {c: (lg[part.center.to_numpy() == c], y[part.center.to_numpy() == c]) for c in range(NUM_CLIENTS)}
    m = summarize(per_center)
    m.update({"encoder": a.encoder, "method": a.method, "head": a.head, "seed": a.seed, "split": a.split, "rounds": a.rounds,
              "updates": updates, "ft_steps": a.ft_steps if a.method == "fedavgft" else None, "epochs": a.epochs if a.method in ("pooled", "local") else None, "train_loss_by_round": losses, "git": git_commit(), "seconds": round(secs, 1)})
    json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    if a.method in ("local", "fedavgft", "fedper"):
        torch.save([x.state_dict() for x in models], os.path.join(out_dir, "models_per_center.pt"))
    else:
        torch.save(models[0].state_dict(), os.path.join(out_dir, "model.pt"))
    log("balanced accuracy  pooled {:.4f} | mean of centers {:.4f} | per center {}".format(
        m["bacc_pooled"], m["bacc_mean_of_centers"], " ".join(f"{m[f'bacc_center_{c}']:.3f}" for c in range(NUM_CLIENTS))))
    log(f"saved to {out_dir} ({secs:.0f} s)")
    with open(os.path.join(a.out, "runs_log.csv"), "a", newline="") as f:
        csv.writer(f).writerow([f"{dt.datetime.now():%Y-%m-%d %H:%M}", split_name, f"{a.encoder}frozen_{mname}",
                                a.seed, a.rounds, f"{m['bacc_pooled']:.4f}", f"{m['bacc_mean_of_centers']:.4f}",
                                round(secs / 60, 1), m["git"], "ok"])


if __name__ == "__main__":
    main()
