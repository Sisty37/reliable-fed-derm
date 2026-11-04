# E3 (calibration) and E6 (unseen centers): choices fixed before running the analyses (4 Nov 2026)

Code: `scripts/mc_dropout.py` (GPU, model A), `scripts/e3_e6_analysis.py` (CPU, models A and B).

## E3

1. **Models:** the E2 models (model A: FedAvg, FedProx, SCAFFOLD, FedBN, FedAvg-FT, pooled,
   local-only; model B: the same without FedBN), image-level and lesion-grouped splits, seeds 0–4.
2. **Temperature:** each center's validation lesions (NLL), global temperature = mean weighted by
   validation images, as in E5; applied to all test predictions of that run. Personalized methods use
   the same rule on their own (per-center-model) predictions.
3. **Metrics (Section 7):** top-label ECE with 15 equal-width bins (also before temperature scaling),
   class-wise ECE (mean over the classes present of the one-vs-rest ECE, 15 bins), Brier score; ECE per
   center. Centers 4 and 5 have fewer than 200 test images and their ECE is flagged as imprecise.
4. **Leakage contrasts (secondary):** for FedAvg and pooled, (i) ECE on the new-lesion subset of the
   image-level test set minus ECE on the full image-level test set (same model), (ii) ECE of the
   image-level model on its test set minus ECE of the lesion-grouped model on its test set; lesion
   bootstrap, 1,000 replicates, seeds averaged inside each replicate. ECE on a subset is not reweighted
   by center (unlike the Gate 2 accuracy measure); this is descriptive.
5. **Conformal coverage under leakage (secondary, Section 3's "leaky calibration"):** FedAvg, LAC,
   α = 0.1; image-level test sets split 100 times into lesion halves; global and client-wise
   thresholds from the calibration half; coverage on evaluation images whose lesion is in training vs
   the others. The lesion-grouped models give the clean reference.
6. **MC-Dropout (model A, FedAvg):** EfficientNet-B0's only dropout layer (p = 0.2, before the
   classifier) is sampled 20 times on the pooled features of the plain view; BatchNorm and
   drop-connect stay in eval mode. Reported as raw (T = 1) ECE, Brier and BA against raw softmax.
   Model B (frozen encoder, linear head) has no dropout and is not included.
7. **Deviation: no 5-member deep ensemble.** The protocol planned one ensemble per condition from the 5
   seeds. In this study each seed uses its own re-split (seed s → split s), so the 5 seed models never
   share a test set, and an ensemble would need 4 extra trainings per condition on one split. The
   ensemble is dropped; this is reported as a deviation.

## E6

8. For each held-out center h and seeds 0–2: balanced accuracy (classes present) and ECE on center h's
   lesion-grouped test set from the leave-one-center-out model, against the lesion-grouped FedAvg model
   of the same seed that trained with center h. Temperature as in item 2 (the held-out center's
   validation lesions take part in the global temperature).
9. The PAD-UFES-20 stress test (E6, second part) follows separately.
