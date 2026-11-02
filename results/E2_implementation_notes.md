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

## Analysis choices (3 Nov 2026, fixed before `scripts/e2_analysis.py` was run on the E2 results)

1. **Ranking metric.** Balanced accuracy on the pooled test set, the FLamby benchmark metric already
   used in E0 and E1; mean-of-centers balanced accuracy is reported as a secondary ranking.
   *Disclosure:* while the runs were still going (2 Nov, 21:35), the per-run balanced accuracies of
   seeds 3 and 4 printed in the training log were looked at, both pooled and mean-of-centers. The
   choice of the pooled metric follows E0/E1 and the protocol's "balanced accuracy (as FLamby)"; both
   rankings are reported in full.
2. **Methods ranked:** FedAvg, FedProx, SCAFFOLD, FedBN, FedAvg-FT, pooled, local-only (7). Each
   split type's ranking uses the mean over seeds 0–4.
3. **Noise floor (Section 9).** τ between single-seed rankings of different seeds within one split
   type (10 pairs × 2 split types), compared with τ between single-seed rankings across split types
   (25 pairs); the seed-mean τ between split types is reported as well. Intervals: lesion bootstrap,
   2,000 replicates, one replicate shared by all methods and seeds.
4. **H3b (interim).** As in the protocol, without FedPer (E7): personalized group = local-only,
   FedAvg-FT, FedBN. The final H3b test is repeated once FedPer exists.
5. **Dose-response.** Weights of the weighted least-squares slope = each center's new-lesion test
   images (seed mean); leakage rate = share of the center's image-level test images whose lesion is
   in training.
6. The analysis was checked on synthetic predictions with planted effects (local-only and the
   personalized methods made to gain from leaked lesions); it recovered the planted ordering.

## Model B (PanDerm-Base, frozen) in E2 (3 Nov 2026, before any model B E2 run)

Code: `scripts/train_linear.py` (`--method fedprox | scaffold | fedavgft | local`). Same embeddings,
centring, loss and U = 6400 local updates per round as in E1 (`results/linear_updates_panderm.json`).

1. **Methods:** FedAvg, FedProx (μ 0.001), SCAFFOLD (server lr 1.0), FedAvg-FT, pooled, local-only;
   FedBN does not apply (no BatchNorm in a linear head on a frozen ViT). Seeds 0–4, both split types.
   FedAvg and pooled seeds 0–2 are reused from E1.
2. **FedAvg-FT:** from the finished FedAvg head, 3 local SGD updates per client (lr 0.01, batch 64,
   random augmented view), each center evaluated with its own head.
3. **Local-only:** the pooled recipe (Adam lr 5e-4, batch 64, 20 epochs) on one center's training
   rows, one head per center, each center evaluated with its own head.
4. The code changes leave FedAvg and pooled unchanged (same random draws), so seeds 3–4 are
   comparable with E1's seeds 0–2.
5. Analysis: `scripts/e2_analysis.py --prefix pandermfrozen_` with the same choices as model A, except
   that FedBN is absent (6 methods) and the H3b personalized group is local-only and FedAvg-FT.
