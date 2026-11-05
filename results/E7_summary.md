# E7: personalization vs global and the final H3b test (6 Nov 2026)

Runs: `scripts/run_e7.ps1` at commit 25e3e06 (tag `e7-prerun`, pushed before the runs); 10 FedPer runs
per model (image-level and lesion-grouped splits, seeds 0–4). Every run logged shared-parameter
difference 0 and local-parameter difference > 0. Analyses: `e2_analysis.py --final`,
`e3_e6_analysis.py --final`, `e5_conformal.py --run-name <m> --n-boot 200`. Outputs:
`results/E7_analysis.json`, `E7_analysis_panderm.json`, `E7_calibration*.json`, `E7_conformal_*.json`.
(The conformal script prints generic labels "E5 models" and "POST HOC on E5 models"; here they refer
to the E7 personalized models, and every conformal result in this file is descriptive.)

## 1. H3b, final pre-specified test (centers 0, 2, 3; inflation in BA points)

| Method | Model A | Model B |
|---|---|---|
| FedAvg | +4.29 | +6.08 |
| FedProx | +4.38 | +5.85 |
| SCAFFOLD | +3.91 | +5.98 |
| FedBN | +4.92 | — |
| FedAvg-FT | +4.79 | +6.27 |
| FedPer (B: FedPer-style head) | +6.72 | **+10.62** |
| Local-only | **+13.72** | +4.15 |
| Pooled (not in the group) | +12.06 | +2.67 |
| **Personalized mean − FedAvg** | **+3.25 [+1.97, +4.55]** | **+0.93 [−0.17, +2.08]** |
| **H3b** | **supported** | **not supported** |

The interim E2 result (without FedPer) was +3.52 [+2.26, +4.75] for model A and −0.87 [−1.85, +0.24]
for model B; adding FedPer does not change either conclusion.

**Reading.** Inflation does not follow "personalized vs global"; it follows how closely a model can
fit its training lesions:
- Model A (full fine-tuning): the methods that train the whole network hard on center data inflate
  most (local-only +13.7, pooled +12.1, both 20 Adam epochs), FedPer is in between (+6.7), and the
  methods that only add local BatchNorm or 3 local steps (FedBN, FedAvg-FT) are at FedAvg's level.
- Model B (frozen encoder): a linear head cannot memorize much, so local-only and pooled inflate
  *least*; the one model B variant with extra capacity, the 256-unit FedPer-style head (training loss
  ≈ 0.10 after 9 rounds), inflates most (+10.6, dose-response slope +18.9 [+15.8, +22.7] pts per
  100% leakage, the steepest of all model B methods).
So H3b holds for model A, but the mechanism is capacity/memorization rather than personalization
as such. This is an interpretation across methods, not a pre-specified test.

## 2. Rankings with FedPer (supplementary; the 7-method E2 rankings remain the RQ2 result)

| | Model A, pooled BA | Model B, pooled BA |
|---|---|---|
| τ between split types | +0.79 [+0.57, +0.93] | **+0.05 [−0.24, +0.24]** |
| single-seed τ across vs noise floor | +0.66 vs +0.72 | +0.11 vs +0.74 |
| FedPer rank, image → lesion | 3 → 2 | **1 → 7** |

- **Model B FedPer-style head is the clearest ranking reversal in the study:** the best method under
  image-level evaluation (0.656) and the worst under lesion-grouped evaluation (0.502).
- **Model A FedPer is the best federated method under clean evaluation:** lesion-grouped pooled BA
  0.555 vs FedAvg 0.495 (+6.0), behind only centralized pooled training (0.566); mean of centers 0.540
  (rank 4).
- Model A local-only again falls from rank 3 to 7 on mean-of-centers BA.

## 3. Calibration (temperature-scaled ECE; raw in brackets)

| | Image-level | Lesion-grouped |
|---|---|---|
| Model A FedPer | 0.042 (0.053) | 0.026 (0.029) |
| Model B FedPer-style | **0.085 (0.066)** | 0.022 (**0.141**) |

- Model A FedPer has the lowest ECE of the federated methods on both splits; per-center ECE 0.03–0.12.
- Model B FedPer-style head repeats the E3 pattern seen for model A pooled/local-only: very
  overconfident on new lesions (raw ECE 0.141), well fixed by temperature scaling on the clean split
  (0.022), but on the leaky image-level split temperature scaling *raises* ECE (0.066 → 0.085). The
  "clean validation, leaky test" miscalibration therefore appears in three memorizing models
  (A pooled, A local-only, B FedPer-style) and in none of the low-capacity ones.

## 4. Conformal coverage on personalized lesion-grouped models (LAC, α = 0.1; descriptive)

| Model / method | Ref. | Lowest center mean coverage: Global / Client-wise / EB-ShrinkFCP | Worst-center q10: ref → EB | EB − ref [95% CI, 200 reps] |
|---|---|---|---|---|
| A FedPer | global | 0.859 / 0.900 / 0.900 | 0.826 → 0.834 | +0.80 [−0.98, +3.40] |
| A FedBN | client-wise | 0.837 / 0.899 / 0.888 | 0.808 → 0.837 | +2.92 [−1.29, +4.78] |
| A FedAvg-FT | client-wise | 0.796 / 0.899 / 0.893 | 0.790 → 0.838 | +4.74 [−0.26, +6.18] |
| A Local-only | client-wise | 0.865 / 0.899 / 0.896 | 0.813 → 0.801 | −1.20 [−3.93, +1.77] |
| B FedPer-style | client-wise | 0.822 / 0.900 / 0.899 | 0.815 → 0.837 | +2.25 [−0.84, +3.68] |
| B FedAvg-FT | client-wise | 0.760 / 0.898 / 0.898 | 0.800 → 0.804 | +0.42 [−2.22, +2.52] |
| B Local-only | global | 0.869 / 0.897 / 0.898 | 0.831 → 0.857 | +2.58 [−0.32, +3.28] |

- **Personalization does not make a global conformal threshold safe:** global calibration leaves
  some center at 0.76–0.87 mean coverage for every personalized model.
- Client-wise and EB-ShrinkFCP keep every center's mean coverage at 0.89–0.90 (FedBN: 0.888).
- EB-ShrinkFCP improves the worst-center tail in 6 of 7 settings (−1.2 to +4.7 points), never with
  an interval above zero at 200 replicates; fixed-k ShrinkFCP fails the mean-coverage guard in all 7.
  This is consistent with, and only supportive of, the pre-specified 5b result (model A met,
  model B not met).
