# Pre-specified Study Protocol

**Version:** 0.7 (28 October 2026). Version 0.1 is locked at git tag `v0.1-protocol` (commit 5ca4821); changes since then are listed in Section 12. No experiment has been run yet, so no result has informed these changes. Each version is time-stamped by its git tag (`v0.1-protocol` … `v0.7-protocol`).
**Study:** Revisiting Federated Skin Lesion Benchmarks: Lesion-Level Leakage, Calibration and Conformal Coverage on Real Centers.

Any change after the tag is recorded in Section 12 with date and reason. Results obtained before a change are kept and reported.

## 1. Research questions

- **RQ1** (mother paper FW1, FW2, FW5): On six real Fed-ISIC2019 centers with a lesion-grouped split, how reliable are federated accuracy, calibration and conformal coverage, overall, per center, for melanoma, on unseen centers, and under a domain shift to smartphone images?
- **RQ4 (method):** Does shrinking each center's conformal threshold toward the federated global threshold, in proportion to how little calibration data the center has, reduce the worst per-center coverage shortfall without inflating prediction sets?
- **RQ2** (prerequisite): Centralized studies show that lesion leakage can inflate dermoscopy metrics, with mixed evidence on its size. How unevenly is leakage distributed across the Fed-ISIC2019 centers, and how much does it change federated method rankings and per-center accuracy, calibration and conformal coverage?
- **RQ3**: Does personalization improve per-center calibration and coverage, or only accuracy? Does the qualitative conclusion of FENDA-FL (personalized methods under-perform on Fed-ISIC2019) hold under a lesion-grouped split?
  - **H3b:** local and personalized models gain most from leakage, because a leaked lesion's sibling images sit in the same center (92.7% of center 0's test images are leaked). Operationalized with the primary measure computed per method and per center: inflation(m, c) = balanced accuracy of method m on center c's image-level test set minus its reweighted balanced accuracy on center c's new-lesion subset. H3b is supported if the mean inflation of local-only, FedAvg-FT, FedBN and FedPer exceeds that of FedAvg, with the bootstrap interval of the difference above zero, in center 0 (the only center with enough new-lesion images); other centers are descriptive.

## 2. Data

- Fed-ISIC2019 from FLamby (commit edacf54), 23,247 images, 6 centers, 8 classes; images resized and colour-normalized with FLamby's scripts.
- Lesion IDs from the HAM10000 metadata and `ISIC_2019_Training_Metadata.csv`. **Check before E0** (`scripts/check_lesion_ids.py`): per center, images per lesion and the number of IDs that equal the image name. For any center whose IDs are per-image placeholders, the leakage rate is reported as a lower bound and that center is excluded from per-center leakage claims. **Result (27 Oct 2026):** no placeholder IDs in any center (0 of 23,247 IDs equal the image name); 11,847 lesions in total. Images per lesion: center 0 (BCN) 3.47, center 1 (Vidir Molemax) 1.00, center 2 1.98, center 3 1.46, center 4 (MSK) 1.02 (15 lesions with 2+ images), center 5 1.58. The low leakage of centers 1 (0.1%) and 4 (3.0%) therefore reflects genuinely few repeat images per lesion, not missing IDs. Same-lesion images recorded under different IDs cannot be excluded; the perceptual-hash and embedding check addresses this. **ID provenance:** center 0 IDs are `BCN_*` and center 4 IDs `MSK*` (both from `ISIC_2019_Training_Metadata.csv`); centers 1, 2, 3 and 5 are `HAM_*` from the HAM10000 metadata; no ID was missing, so the `IMG_` fallback in `audit_fedisic_leakage.py` was never used. **HAM10000 cross-check:** centers 1, 2, 3 and 5 contain 10,015 images and 7,470 unique lesions (11,847 − 3,576 BCN − 801 MSK), exactly the published HAM10000 counts. **MSK check (done, 27 Oct 2026):** for center 4, 0.0% of lesion IDs carry the same number as their image (e.g. image ISIC_0012669 → lesion MSK4_0011183), and the 819 images map to 801 lesion IDs; the IDs are therefore an independent lesion numbering, not per-image placeholders, and center 4's 3.0% leakage is taken at face value.
- PAD-UFES-20 is a **domain-shift stress test** (smartphone clinical images, not dermoscopy, not an unseen center). Six of the eight classes are shared (MEL, NV, BCC, AK, SCC, BKL; SEK→BKL, ACK→AK). Model probabilities are restricted to these six classes and renormalized. Patient-grouped use only.
- **Pretraining overlap:** model B (PanDerm) was pretrained on about 2.1 million skin images including ISIC 2024 tiles; ISIC 2019, HAM10000, BCN20000 and PAD-UFES-20 are not listed among its pretraining sources. "Not listed" is not the same as "not seen": before E1 we check the PanDerm supplementary material and ask the authors. If ISIC 2019 or HAM10000 images were used, model B is reported as possibly contaminated and analysis 2f relies on the model A contrast only. Models pretrained on PAD-UFES-20 (e.g. MedSigLIP) are excluded.

## 3. Splits (files in `splits/`, checksums in `splits/SHA256SUMS.txt`)

| Name | Definition | Use |
| --- | --- | --- |
| official | FLamby fixed split (image-level, stratified by center) | **E0:** reproduce published results |
| resplit_image_seed{0–4} | 20% test per center, image-level, stratified by class | Leaky condition |
| resplit_lesion_seed{0–4} | 20% test per center, lesion-grouped, class quotas; cross-center lesions kept in train | Clean condition |

- **Validation:** within each training set, 10% of lesions are held out **lesion-grouped in both conditions**; they are used for temperature scaling and, where needed, checkpoint choice. (A leaky-validation variant was considered and dropped: an image-level validation set must be removed from training, which changes the training set and needs retraining. Leaky calibration is already captured, because the conformal calibration half of the image-level test set shares lesions with training.)
- **Conformal calibration/evaluation:** within each test set, halves are drawn by lesion, repeated 100 times.
- **New-lesion subset:** for models trained on image-level splits, test images whose lesion does not appear in training. Balanced accuracy is the mean of per-class recalls, so the class mix itself does not matter; what matters is the center mix within each class. **Reweighting:** for each class, recall on the new-lesion subset is the weighted mean of per-center recalls, with weights equal to each center's share of that class in the full image-level test set (renormalized over centers that have new-lesion images of that class). A class with fewer than 5 new-lesion images (pooled over centers) is dropped from **both** terms of the inflation. **Remaining threat:** new-lesion images are mostly single-image lesions, which may differ from multi-image lesions even after reweighting (e.g. in difficulty or follow-up status); this is stated as a limitation. **Sensitivity check:** the between-condition difference (balanced accuracy of the image-level model on its own test set minus that of the lesion-grouped model on its own test set) is reported alongside, and we state whether both point the same way.

## 4. Models and training

| Setting | Value |
| --- | --- |
| Model A (main, anchor) | EfficientNet-B0, ImageNet pretrained, full fine-tuning (comparable with published Fed-ISIC2019 results) |
| Model A-frozen (control for 2f) | Same EfficientNet-B0 and ImageNet weights, frozen; federated linear head. To keep augmentation comparable, training embeddings are computed for 5 augmented views per image with FLamby's training transforms; validation and test use the plain view. Remaining differences from model A besides the regime: a fixed set of 5 views instead of fresh augmentation each step, and a linear head instead of the full network |
| Model B (main, foundation model) | PanDerm-Base (ViT-B/16, dermatology foundation model, CC-BY-NC-ND), frozen; embeddings as for A-frozen (5 augmented views for training); federated linear head; personalization = client-specific heads (FedPer-style) |
| Linear heads (A-frozen, B) | SGD lr 0.01, batch 64, same rounds and local updates as model A. **Convergence check** on training loss only: if the pooled training loss falls by more than 1% over the last round, local updates per round are doubled and the run repeated; the rule is applied identically to both split types |
| Model C (ablation only, if time) | DINOv3 ConvNeXt-Tiny, full fine-tuning; E1 and E2 (FedAvg, FedProx, FedAvg-FT), 3 seeds. FedBN not applicable (LayerNorm) |
| Loss | Weighted focal loss, γ = 2, FLamby class weights |
| Federated optimizer | SGD, lr 0.01, batch 64, 100 local updates per round, 9 rounds |
| FedProx μ | 0.001 |
| SCAFFOLD | SGD lr 0.01, server lr 1.0 |
| Fine-tuning (FedAvg-FT) | 3 local steps |
| Pooled (centralized) | Adam, lr 5e-4, batch 64, 20 epochs |
| Local-only | Same as pooled, per center |
| Initialization | ImageNet only; no warm start from pooled models |
| Rounds | 9 (FLamby). **27 rounds** for model A FedAvg is part of E1, because under-trained models may memorize less and understate leakage |
| Seeds | 5 (E2, E7), 3 (E0, E0b, E1, E4, E6) |

**FLamby settings verified** against `flamby/datasets/fed_isic2019/common.py` and `flamby/config_isic2019.json` (commit edacf54): batch 64, pooled Adam lr 5e-4 for 20 epochs, `get_nb_max_rounds(100)` = 9; FedAvg, FedProx (μ 0.001), SCAFFOLD (server lr 1.0) and FedAvg-FT (3 steps) with SGD lr 0.01. No hyperparameter is tuned on any test set.

## 5. Calibration and conformal prediction

- Temperature scaling per client on its validation lesions; global temperature = sample-weighted mean.
- Split conformal with LAC and APS scores; target coverage 90% (main), also 80% and 95%.
- Aggregation: global (pooled scores), client-wise (each center's own quantile), sample-weighted quantile; baselines FCP (Lu et al., 2023) and personalized FCP (Min et al., 2025).
- **Proposed: shrinkage federated conformal calibration (ShrinkFCP, RQ4).** For center c with n_c calibration lesions' scores, let q_c be its own conformal quantile (level ⌈(n_c+1)(1−α)⌉/n_c) and q_g the global quantile obtained federatedly (each client shares a 1,000-bin histogram of its scores, as in FCP-style aggregation; no raw scores leave the client). The threshold is q̂_c = w_c·q_c + (1 − w_c)·q_g with w_c = n_c/(n_c + k). k = 0 gives client-wise and k → ∞ gives global calibration. **k is chosen once on validation lesions only** from {0, 10, 25, 50, 100, 200, 500, ∞}, minimizing the largest per-center coverage shortfall max_c max(0, (1−α) − cov_c), ties broken by smaller mean set size; the chosen k is then fixed for all test evaluations. Shrinkage does not carry a finite-sample coverage guarantee per center; this is stated and coverage is assessed empirically.
- **Success criterion (RQ4, fixed now):** on lesion-grouped models A and B at α = 0.1, ShrinkFCP reduces the worst-center coverage shortfall by at least 2 points compared with the better of global and client-wise calibration, with the lesion-bootstrap interval of the reduction above zero, while mean set size grows by no more than 10%. FCP and personalized FCP are reported as comparisons. If the criterion is not met, the result is reported as negative.
- PAD-UFES-20: (i) transfer, using the Fed-ISIC2019 lesion-grouped calibration scores restricted to the six shared classes; (ii) recalibrated on 50% of PAD-UFES-20 patients and evaluated on the other 50%, repeated 100 times.

## 6. Experiments

| ID | Content | Runs (model A; models A-frozen and B take minutes each) | RQ |
| --- | --- | --- | --- |
| E0 | **Official-split reproduction:** model A FedAvg, FedProx, SCAFFOLD, pooled; compared with FLamby and FENDA-FL published balanced accuracy | 12 | Anchor check |
| E0b | **Mother paper's data setting:** HAM10000 (centers 1, 2, 3, 5; 10,015 images, 7,470 lesions) split 60/20/20 by lesion as in FedHybridXAI (p. 293), class balancing on the training set only (p. 294), three label-skewed clients (p. 298 does not state how they were built, so we use a Dirichlet α = 0.5 class partition). Lesion-grouped split (as in the mother paper) vs image-level split; model A FedAvg. This shows how much that data setting would change **without** lesion grouping; it does not re-evaluate the mother paper's results. FedHybridXAI itself is not re-run (no code released; XAI is outside this study's scope) | 6 | RQ2, link to mother paper |
| E1 | FedAvg + pooled; image-level vs lesion-grouped; models A, A-frozen, B; plus model A FedAvg at 27 rounds | 12 + 6 (27 rounds) | RQ2, Gate 2 |
| E2 | FedAvg, FedProx, FedBN (A only), SCAFFOLD, FedAvg-FT, pooled, local-only; both split types; models A and B | 70 | RQ2 |
| E3 | Softmax, MC-Dropout (20 passes), one 5-member deep ensemble per condition (from the 5 seeds; its uncertainty comes from the lesion bootstrap only), conformal; on E2 models | post hoc | RQ2, RQ1 |
| E4 | Balancing/augmentation applied to training only vs to all splits | 6 | RQ2 |
| E5 | Conformal aggregation strategies on lesion-grouped models A and B | post hoc | RQ1 |
| E5b | **ShrinkFCP** vs global, client-wise, FCP, personalized FCP on lesion-grouped models A and B; α = 0.05, 0.1, 0.2; k chosen on validation lesions; also on leave-one-center-out (E6) models | post hoc | RQ4 |
| E6 | Leave-one-center-out (lesion-grouped models); PAD-UFES-20 stress test | 18 | RQ1 |
| E7 | Personalization vs global (A: FedBN, FedAvg-FT, FedPer; B: client-specific heads; personalized FCP): accuracy, ECE, coverage per center; subgroups; both split types; H3b | 10 (FedPer) + E2 | RQ3 |
| E8 | Model C if time; calibration-set size | about 12 | Ablation |

Model A total: about 140 runs.

**Analysis 2f (RQ2):** does the inflation depend on the adaptation regime? **Primary contrast:** model A (full fine-tuning) vs model A-frozen (same architecture and pretraining). **Secondary:** model B, which also differs in architecture and pretraining data, so a difference there is not attributed to memorization alone.

## 7. Metrics

- Balanced accuracy (as FLamby), macro-F1, AUROC, melanoma recall; ECE (15 equal-width bins), class-wise ECE, Brier score; coverage (marginal, per center, per class, per subgroup), average set size, singleton rate, rate of sets containing both melanoma and nevus; Kendall's τ between method rankings.
- **Absent classes:** per-center balanced accuracy, macro-F1 and macro-AUROC are averaged over the classes present in that center's test set; the class list per center is reported.
- **Small centers:** ECE for centers with fewer than 200 test images (centers 4 and 5) is reported with its bootstrap interval and flagged as imprecise.

## 8. Reporting rules from the sample-size analysis

- Per-center coverage is reported for all six centers with 95% intervals.
- Melanoma coverage is reported pooled and for center 0; **centers 2, 3 and 4 are labelled indicative**; centers 1 and 5 (4 and 9 melanoma test images in the official split) are not reported per center.
- Class-conditional results are reported only for classes present in a center's test set.

## 9. Statistics

- **Intervals:** 95% intervals by lesion-wise cluster bootstrap (2,000 resamples of test lesions). For a contrast within the same model and test set (e.g. reported vs new-lesion performance), the same resampled lesions are used for both terms; seeds are averaged inside each bootstrap replicate. **Limitation:** these intervals reflect test-set sampling only, not variation between training runs; the seed-to-seed range is reported next to every interval.
- **No signed-rank tests on seeds:** with 5 paired seeds the smallest two-sided Wilcoxon p-value is 2/32 = 0.0625 (0.25 with 3 seeds), so seed-level tests cannot reach 0.05. Decisions use effect sizes with bootstrap intervals; seed-to-seed spread is reported descriptively.
- **Primary outcome (one):** see Gate 2. All other comparisons are secondary, reported as effect sizes with intervals, without significance claims.
- **Rankings:** Kendall's τ between the method rankings of the two split types is compared with a **noise floor**, the τ between rankings from different seeds within the same split type; τ intervals by bootstrap over lesions.
- **Meaningful difference (fixed now):** at least 2 percentage points, with the 95% interval excluding zero.

## 10. Decision gates

- **Gate 1 (after E0):** reference value = FENDA-FL's FedAvg global-model balanced accuracy on the Fed-ISIC2019 test set, **0.657** (EfficientNet-B0, FLamby split). Gate 1 passes if model A FedAvg on the official split (mean of 3 seeds, pooled test set, balanced accuracy) is within ±5 points of 0.657. FLamby's own published FedAvg value is recorded next to it before E0. If Gate 1 fails, the pipeline is debugged before E1.
- **Gate 2 (after E1), primary outcome:** inflation of model A FedAvg (9 rounds) = balanced accuracy on its own image-level test set minus reweighted balanced accuracy on the new-lesion subset of the same test set, same model; interval as in Section 9. If the inflation is at least 2 points with the interval above zero, RQ2 is the headline. Otherwise the headline is RQ1 and RQ2 is reported as a small or null effect with its interval. Melanoma recall, coverage, ECE, the between-condition difference and the 27-round result are secondary.
- **Scope gate (after E5):** if time is short, E7 and E8 are reduced; E0, E1, E2, E5, E5b are never dropped.

## 11. Integrity, registration and release

- `tests/test_splits.py`: no lesion in both train and test of lesion-grouped splits; image-level and lesion-grouped train sizes within 1%; full splits cover the same 23,247 images.
- Split files are never edited by hand; checksums verified before each experiment batch.
- All runs are logged, including failed and negative ones.
- The protocol is not registered externally; its versions are fixed by git tags, and the repository (with full history) is released with the paper. Later changes are logged here.
- Release: code, configurations, split files (lesion IDs dropped if the ISIC licence requires). PanDerm-derived heads or embeddings are released only if the PanDerm authors confirm this is allowed under CC-BY-NC-ND.

## 12. Deviations log

| Date | Change | Reason |
| --- | --- | --- |
| 27 Oct 2026 | RQ2 reworded to state that centralized leakage is prior evidence and to focus on the federated, per-center effect; analysis 2e added (per-center effect vs per-center leakage rate) | Literature round 3 found 2026 centralized leakage studies (incl. one with no AUC effect); no data analysed |
| 27 Oct 2026 | RQ3 extended to re-test FENDA-FL and FedOPAL conclusions; FedPer added to E7 and E7 run on both split types | Two Fed-ISIC2019 personalization studies found in round 3; no data analysed |
| 27 Oct 2026 | Backbones: model B (PanDerm-Base, frozen, federated linear head) added to E1, E2, E5, E7; former E8 backbones (ResNet18, MobileNetV3, ViT-Tiny) replaced by DINOv3 ConvNeXt-Tiny as optional ablation; analysis 2f added; pretraining-overlap rule added (MedSigLIP excluded) | Reviewer expectation of recent models; foundation model allows a full-fine-tuning vs frozen comparison within 6 GB GPU; no data analysed |
| 27 Oct 2026 (v0.4) | Wilcoxon tests on seeds dropped; one primary outcome defined for Gate 2 (within-model inflation on the reweighted new-lesion subset, lesion bootstrap); Gate 1 added; E0 (official-split reproduction) and E0b (mother-paper setting) added; model A-frozen control added for 2f; 27 rounds moved into E1; lesion-grouped validation stated for both conditions and a leaky-validation variant added; new-lesion reweighting; lesion-ID check; SCAFFOLD added to E2; RQ3 reworded to the qualitative FENDA-FL conclusion and H3b added; τ noise floor; absent-class and small-center metric rules; centers 2–4 indicative; PAD-UFES-20 defined as a domain-shift stress test with 6-class restriction and two calibration sources; ensemble variance clarified; OSF registration and release rule added | External methodological review of v0.3; no data analysed |
| 27 Oct 2026 (v0.5) | Leaky-validation variant dropped (would need retraining); new-lesion reweighting defined per class over centers, with a drop rule, a stated threat and a between-condition sensitivity check; E0b reworded and specified (mother paper pp. 293, 294, 298; Dirichlet α = 0.5 clients, as the mother paper does not state its partition); lesion-ID provenance, HAM10000 cross-check (7,470 lesions) and MSK check added; A-frozen and B use 5 augmented embedding views and a pre-specified convergence rule; Gate 1 reference fixed at 0.657 (FENDA-FL); E0b seeds stated; H3b operationalized; bootstrap limitation stated | Second external methodological review; no data analysed |
| 27 Oct 2026 (v0.6) | Planned OSF registration dropped; versions are time-stamped by git tags and the full history is released with the paper | Author's decision; no data analysed |
| 28 Oct 2026 (v0.7) | RQ4 and experiment E5b added: shrinkage federated conformal calibration (ShrinkFCP) with k chosen on validation lesions and a fixed success criterion | A method contribution addressing the small-center calibration problem the study is designed to measure; post hoc, no extra training; no data analysed |
