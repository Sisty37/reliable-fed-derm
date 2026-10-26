# Pre-specified Study Protocol

**Version:** 0.2 (27 October 2026). Version 0.1 is locked at git tag `v0.1-protocol` (commit 5ca4821); changes since then are listed in Section 12. No experiment has been run yet, so no result has informed these changes.
**Study:** Revisiting Federated Skin Lesion Benchmarks: Lesion-Level Leakage, Calibration and Conformal Coverage on Real Centers.

Any change after the tag is recorded in Section 12 with date and reason. Results obtained before a change are kept and reported.

## 1. Research questions

- **RQ1** (mother paper FW1, FW2, FW5): On six real Fed-ISIC2019 centers with a lesion-grouped split, how reliable are federated accuracy and conformal coverage, overall, per center, for melanoma, and on unseen centers?
- **RQ2** (prerequisite): Centralized studies show that lesion leakage can inflate dermoscopy metrics, with mixed evidence on its size. How unevenly is leakage distributed across the Fed-ISIC2019 centers, and how much does it change federated method rankings and per-center accuracy, calibration and conformal coverage?
- **RQ3**: Does personalization improve per-center calibration and coverage, or only accuracy, and do published conclusions about personalization on Fed-ISIC2019 (FENDA-FL, FedOPAL) hold under a lesion-grouped split?

## 2. Data

- Fed-ISIC2019 from FLamby (commit edacf54), 23,247 images, 6 centers, 8 classes; images resized and colour-normalized with FLamby's scripts.
- Lesion IDs from `ISIC_2019_Training_Metadata.csv` (all 23,247 images have an ID).
- PAD-UFES-20 (unseen domain), mapped to the shared classes MEL, NV, BCC, AK, SCC, BKL (SEK→BKL, ACK→AK); patient-grouped use only.

## 3. Splits (files in `splits/`, checksums in `splits/SHA256SUMS.txt`)

| Name | Definition | Use |
| --- | --- | --- |
| official | FLamby fixed split (image-level, stratified by center) | Reproduce published setting |
| resplit_image_seed{0–4} | 20% test per center, image-level, stratified by class | Leaky condition |
| resplit_lesion_seed{0–4} | 20% test per center, lesion-grouped, class quotas; cross-center lesions kept in train | Clean condition |

Within each training set, 10% of lesions are held out (lesion-grouped) as validation for early stopping and temperature scaling. Within each test set, conformal calibration and evaluation halves are drawn by lesion, repeated 100 times.

**New-lesion subset:** for models trained on image-level splits, the test images whose lesion does not appear in training are evaluated separately. This is the real-data analogue of "new lesions" in the mechanism simulation.

## 4. Models and training

| Setting | Value (FLamby defaults unless noted) |
| --- | --- |
| Backbone | EfficientNet-B0, ImageNet pretrained (main); ResNet18, MobileNetV3, ViT-Tiny in E8 only |
| Loss | Weighted focal loss, γ = 2, FLamby class weights |
| Federated optimizer | SGD, lr 0.01, batch 64, 100 local updates per round, 9 rounds |
| FedProx μ | 0.001 |
| Fine-tuning (FedAvg-FT) | 3 local steps |
| Pooled (centralized) | Adam, lr 5e-4, 20 epochs |
| Local-only | Same as pooled, per center |
| Initialization | ImageNet only; no warm start from pooled models |
| Rounds sensitivity | 27 rounds (E8) because FLamby notes the round cap may limit convergence |
| Seeds | 5 (main), 3 (E1, E4, E6) |

No hyperparameter is tuned on any test set.

## 5. Calibration and conformal prediction

- Temperature scaling per client on its validation lesions; global temperature = sample-weighted mean.
- Split conformal with LAC and APS scores; target coverage 90% (main), also 80% and 95%.
- Aggregation: global (pooled scores), client-wise (each center's own quantile), sample-weighted quantile; baselines FCP (Lu et al., 2023) and personalized FCP (Min et al., 2025).

## 6. Experiments

| ID | Content | RQ |
| --- | --- | --- |
| E1 | FedAvg + pooled; image-level vs lesion-grouped; 3 seeds | RQ2, Gate 2 |
| E2 | FedAvg, FedProx, FedBN, FedAvg-FT, pooled, local; both split types; 5 seeds | RQ2 |
| E3 | Softmax, MC-Dropout (20 passes), 5-member ensembles, conformal on E2 models | RQ2, RQ1 |
| E4 | Balancing/augmentation applied to training only vs to all splits | RQ2 |
| E5 | Conformal aggregation strategies on lesion-grouped models | RQ1 |
| E6 | Leave-one-center-out; PAD-UFES-20 | RQ1 |
| E7 | Personalization vs global (FedBN, FedAvg-FT, FedPer, personalized FCP): accuracy, ECE, coverage per center; subgroups; both split types | RQ3 |
| E8 | Backbones, 27 rounds, calibration-set size | Ablation |

## 7. Metrics

Balanced accuracy (primary, as FLamby), macro-F1, AUROC, melanoma recall; ECE (15 equal-width bins), class-wise ECE, Brier score; coverage (marginal, per center, per class, per subgroup), average set size, singleton rate, rate of sets containing both melanoma and nevus; Kendall's τ between method rankings.

## 8. Reporting rules from the sample-size analysis

- Per-center coverage is reported for all six centers with 95% intervals.
- Melanoma coverage is reported pooled and for center 0; centers 2–4 are labelled indicative; centers 1 and 5 (4 and 9 melanoma test images in the official split) are not reported per center.
- Class-conditional results are reported only for classes present in a center's test set.

## 9. Statistics

- 95% intervals by lesion-wise cluster bootstrap (2,000 resamples).
- Paired comparisons (image-level vs lesion-grouped, same seed) by Wilcoxon signed-rank test; Holm correction across metrics.
- **Meaningful difference (fixed now):** at least 2 percentage points in balanced accuracy or coverage, with the interval excluding zero.

## 10. Decision gates

- **Gate 2 (after E1):** if the leaky–clean difference is meaningful for balanced accuracy, melanoma recall or coverage on new lesions, RQ2 is the headline. Otherwise the headline is RQ1 (per-center and melanoma coverage), and RQ2 is reported as a null or small effect with its interval.
- **Scope gate (after E5):** if time is short, E7 and E8 are reduced; E1, E2, E5 are never dropped.

## 11. Integrity checks

- `tests/test_splits.py`: no lesion in both train and test of lesion-grouped splits; image-level and lesion-grouped train sizes within 1%; full splits cover the same 23,247 images.
- Split files are never edited by hand; checksums verified before each experiment batch.
- All runs are logged, including failed and negative ones.

## 12. Deviations log

| Date | Change | Reason |
| --- | --- | --- |
| 27 Oct 2026 | RQ2 reworded to state that centralized leakage is prior evidence and to focus on the federated, per-center effect; analysis 2e added (per-center effect vs per-center leakage rate) | Literature round 3 found 2026 centralized leakage studies (incl. one with no AUC effect); no data analysed |
| 27 Oct 2026 | RQ3 extended to re-test FENDA-FL and FedOPAL conclusions; FedPer added to E7 and E7 run on both split types | Two Fed-ISIC2019 personalization studies found in round 3; no data analysed |
