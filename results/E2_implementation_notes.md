# E2, model A: implementation choices fixed before any E2 training (2 Nov 2026)

Protocol v0.10, Section 6: FedAvg, FedProx, FedBN (model A only), SCAFFOLD, FedAvg-FT, pooled and
local-only; image-level and lesion-grouped re-splits; 5 seeds. Code: `scripts/train_fl.py`
(`--method fedbn`, `--method local`), `fedderm/fl.py` (FedBN aggregation), `scripts/finetune_fedavg.py`
(FedAvg-FT). Points the protocol leaves open:

1. **Splits for seeds 3 and 4.** `splits/e1/resplit_{image,lesion}_seed{3,4}_val.csv`, made with
   `make_e1_splits.py` exactly as seeds 0–2 (10% of each center's training lesions to validation).
   Re-running it for seeds 0–2 reproduces the existing files byte for byte (checked with `git diff`).
2. **Reuse.** The E1 runs for FedAvg (9 rounds) and pooled, seeds 0–2, are reused (Section 6); seeds 3
   and 4 are new. Run seed s always uses split seed s.
3. **FedBN.** As FedAvg, but BatchNorm weights and biases are not averaged (BatchNorm running
   statistics are already local in all methods, as in FLamby). Each center is evaluated with its own
   client's model.
4. **Local-only.** The pooled recipe (Adam lr 5e-4, batch 64, 20 epochs, final model) trained on one
   center's training data, one model per center, each evaluated on its own center's test set; seed
   s + 1000 × center.
5. **FedAvg-FT.** From each finished FedAvg run: global parameters plus each client's own BatchNorm
   statistics (the state each client holds at the end of FedAvg), then 3 local SGD updates (lr 0.01,
   batch 64, FLamby augmentations, weighted focal loss, AMP), as in FLamby's FedAvgFineTuning. Each
   center is evaluated with its own fine-tuned model. FLamby's benchmark instead evaluates client 0's
   fine-tuned model on every center; we use per-center evaluation because that is what
   personalization means and what E7 and H3b compare.
6. **Global methods** (FedAvg, FedProx, SCAFFOLD) are evaluated with the client 0 copy, as in E0 and
   E1; pooled with the pooled model.
7. **Outputs** as before (`metrics.json`, `preds.npz`, `preds_val.npz`); personalized methods save all
   six models in `models_per_center.pt`.
8. **Model B in E2** (FedProx, SCAFFOLD, FedAvg-FT, pooled, local-only linear heads) follows in a
   separate step, with its own notes, before its runs.
