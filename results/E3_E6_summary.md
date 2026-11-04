# E3 (calibration) and E6 (unseen centers): results (4 Nov 2026)

Run with `scripts/mc_dropout.py` then `scripts/e3_e6_analysis.py` (model A) and
`scripts/e3_e6_analysis.py --prefix pandermfrozen_` (model B), choices fixed beforehand in
`E3_E6_implementation_notes.md`. Outputs: `results/E3_E6_analysis.json`,
`results/E3_E6_analysis_pandermfrozen.json`, `runs/resplit_*_val/fedavg_r9_s*/preds_mc*.npz`.
Note: the copy of `mc_dropout.py` run on the study PC computes the dropout passes under
`torch.no_grad()` and also calls `.detach()`; the two are equivalent for inference.

## E3.1 Calibration (test sets, temperature-scaled, mean of 5 seeds)

| Method | Model A image | Model A lesion | Model B image | Model B lesion |
|---|---|---|---|---|
| FedAvg | 0.061 | 0.046 | 0.040 | 0.028 |
| FedProx | 0.062 | 0.032 | 0.039 | 0.029 |
| SCAFFOLD | 0.063 | 0.045 | 0.038 | 0.028 |
| FedBN | 0.048 | 0.034 | — | — |
| FedAvg-FT | 0.050 | 0.039 | 0.040 | 0.026 |
| Pooled | 0.074 (raw 0.024) | 0.018 | 0.032 | 0.030 |
| Local-only | 0.087 (raw 0.031) | 0.019 | 0.042 | 0.033 |

(Top-label ECE, 15 bins. Class-wise ECE and Brier: see the JSON files.)

1. **Centers differ more than methods.** Per-center ECE spans 0.03–0.21 (model A) and 0.03–0.28
   (model B); center 1 (and the small centers 4–5, flagged imprecise) are worst for every method.
   A single pooled ECE hides this.
2. **Temperature scaling on clean validation lesions can harm a leaky test set.** For model A pooled
   and local-only on the image-level split, ECE rises from 0.024 → 0.074 and 0.031 → 0.087 after
   scaling, while on the lesion-grouped split scaling helps (0.052 → 0.018; 0.094 → 0.019).
   Interpretation (not tested separately): the validation fold holds only new lesions, on which these
   fully trained models are overconfident, so the fitted T > 1; applied to a test set where most
   images belong to training lesions, it makes those predictions underconfident. Model B (frozen
   encoder) does not show this (scaling helps on both splits), consistent with its weaker memorization.
3. **MC-Dropout gives no benefit** (model A FedAvg, raw): ECE 0.065 vs 0.068 (image), 0.047 vs 0.047
   (lesion); Brier and BA unchanged. With a single dropout layer before the classifier (p = 0.2) the
   sampled predictions barely differ from the softmax.

## E3.2 Leakage and calibration (ECE points ×100, 95% lesion-bootstrap CI)

| Model | Method | New-lesion minus full (same model) | Image-level minus lesion-grouped model |
|---|---|---|---|
| A | FedAvg | +2.94 [+2.47, +3.90] | +1.52 [+0.44, +2.13] |
| A | Pooled | −1.55 [−1.83, −0.59] | +5.66 [+4.23, +5.67] |
| B | FedAvg | +2.04 [+1.55, +2.86] | +1.16 [+0.08, +1.70] |
| B | Pooled | +2.14 [+1.62, +3.05] | +0.20 [−0.60, +0.85] |

- For FedAvg (both models) the image-level test set **understates ECE on new lesions by 2–3 points**.
- For model A pooled the sign flips, matching item 2: after temperature scaling the leaked images are
  the badly calibrated ones (underconfident), so the full test set looks worse than the new lesions.
- Some point estimates lie at the edge of their interval (A pooled). ECE is biased upward in smaller
  resamples, so the bootstrap distribution is skewed; the intervals are reported as computed and are
  descriptive (secondary analysis).

## E3.3 Conformal coverage under leakage (FedAvg, LAC, α = 0.1, 5 seeds × 100 lesion splits)

| Model | Threshold | Leaked images | New-lesion images | All |
|---|---|---|---|---|
| A | Global | 0.897 | 0.906 | 0.900 |
| A | Client-wise | 0.907 | 0.892 | 0.901 |
| B | Global | 0.899 | 0.903 | 0.901 |
| B | Client-wise | 0.907 | 0.894 | 0.902 |

Lesion-grouped reference: 0.899 for both thresholds and both models.
**Marginal coverage is robust to lesion leakage:** the leaked/new gap is at most 1.5 points, and all
coverage is 0.90 overall. A negative result for the "leaky calibration" concern at this α:
leakage inflates accuracy (E1) and understates ECE (E3.2) but barely moves conformal coverage.

## E6 Unseen centers (lesion-grouped test set of center h; LOCO vs in-federation FedAvg; seeds 0–2)

| Center | A: BA in-fed → unseen (drop) | A: ECE in → out | B: BA in-fed → unseen (drop) | B: ECE in → out |
|---|---|---|---|---|
| 0 | 0.487 → 0.312 (−17.5) | 0.031 → 0.101 | 0.480 → 0.312 (−16.8) | 0.029 → 0.076 |
| 1 | 0.741 → 0.680 (−6.0) | 0.198 → 0.160 | 0.788 → 0.717 (−7.1) | 0.184 → 0.168 |
| 2 | 0.586 → 0.540 (−4.6) | 0.079 → 0.057 | 0.642 → 0.607 (−3.5) | 0.084 → 0.086 |
| 3 | 0.458 → 0.373 (−8.5) | 0.077 → 0.090 | 0.542 → 0.499 (−4.3) | 0.059 → 0.050 |
| 4* | 0.406 → 0.398 (−0.8) | 0.120 → 0.151 | 0.330 → 0.295 (−3.4) | 0.191 → 0.222 |
| 5* | 0.436 → 0.535 (+9.9) | 0.158 → 0.149 | 0.455 → 0.450 (−0.5) | 0.255 → 0.276 |
| Mean | 0.519 → 0.473 (−4.6) | 0.111 → 0.118 | 0.539 → 0.480 (−5.9) | 0.134 → 0.146 |

\* fewer than 200 test images; imprecise (the +9.9 for model A center 5 is within its noise).

- Excluding a center costs about 5 BA points on it on average, and **center 0 loses 17 points with
  either encoder**, i.e. the frozen foundation-model features do not remove the domain gap.
- ECE changes little on average (+0.7 / +1.2 points) but triples for center 0.
- Together with `E6_conformal_summary.md`: on these held-out models global conformal under-covers
  (lowest center mean 0.83 / 0.75), and EB-ShrinkFCP restores ≈ 0.90 for every center.
