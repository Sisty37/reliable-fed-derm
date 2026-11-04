# E7 (personalization vs global, final H3b): choices fixed before the runs (4 Nov 2026)

Code: `fedderm/fl.py` (strategy `fedper`), `scripts/train_fl.py --method fedper` (model A),
`scripts/train_linear.py --encoder panderm --method fedper` (model B), `scripts/run_e7.ps1`;
analyses `scripts/e2_analysis.py --final`, `scripts/e3_e6_analysis.py --final`,
`scripts/e5_conformal.py --run-name <method>`.

## Training

1. **Model A FedPer** (Arivazhagan et al., 2019): EfficientNet-B0, the classifier layer (`_fc`, weight
   and bias) is client-specific and never averaged; every other parameter is averaged as in FedAvg
   (weighted by training images). BatchNorm running statistics stay local, as for all federated runs
   (FLamby convention). Same recipe as FedAvg: SGD lr 0.01, batch 64, 9 rounds × 100 local updates,
   weighted focal loss, AMP. All clients start from the same ImageNet-pretrained model and the same
   initial head. Each center is evaluated with its own client's model (as FedBN).
2. **Model B FedPer-style head** (protocol v0.10): on the centred PanDerm embeddings,
   Linear(768, 256) – ReLU – Linear(256, 8); the hidden layer is federated, the output layer is
   client-specific. Same optimizer, 9 rounds and U as the model B FedAvg linear head
   (`results/linear_updates_panderm.json`). **U is not re-tuned** for the two-layer head; the mean
   client training loss by round is logged and reported. Same 5-view augmentation sampling.
3. **Runs:** both split types (image-level, lesion-grouped, with validation fold), seeds 0–4:
   10 runs per model. Tags `fedper_r9_s<seed>` and `pandermfrozen_fedper_r9_s<seed>`.
4. **Sanity check logged in every run:** largest difference between clients in shared parameters
   (must be 0) and in local parameters (must be > 0).

## Analyses

5. **H3b, final test (pre-specified, Section 1):** `e2_analysis.py --final`. Personalized group:
   model A = local-only, FedAvg-FT, FedBN, FedPer; model B = local-only, FedAvg-FT, FedPer-style head
   (no BatchNorm). Mean inflation of the group minus FedAvg in centers 0, 2, 3, weighted by
   new-lesion counts, lesion bootstrap 2,000 replicates; supported if the interval is above zero.
   The E2 interim result (without FedPer) is kept and reported as interim. Dose-response slopes are
   recomputed with FedPer (secondary). The 8-method rankings printed in this mode are supplementary;
   the E2 7-method rankings remain the RQ2 result.
6. **Calibration:** `e3_e6_analysis.py --final` adds FedPer to the E3 calibration table (same
   temperature rule; ECE, class-wise ECE, Brier, ECE per center).
7. **Coverage per center (descriptive):** `e5_conformal.py --run-name <m>` on the lesion-grouped
   personalized models (FedBN, FedAvg-FT, FedPer, local-only; model B without FedBN), the same
   conformal methods and splits as E5, 200 bootstrap replicates (descriptive, no success criterion).
8. **Not done (deviation already recorded):** personalized FCP (Min et al., 2025) is not
   implemented; client-wise calibration on personalized models is reported in its place.
