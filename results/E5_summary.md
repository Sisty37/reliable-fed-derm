# E5 / E5b: federated conformal prediction on the real centers and ShrinkFCP (RQ4)

Run 3 Nov 2026 with `scripts/e5_conformal.py` (commit tagged `e5-preregistered`, choices in
`results/E5_implementation_notes.md`). Lesion-grouped FedAvg models, seeds 0–4, 100 lesion-wise
calibration/evaluation splits per seed, 2,000 lesion-bootstrap replicates for the RQ4 gain. Full output in
`results/E5_conformal.json` (model A) and `results/E5_conformal_pandermfrozen.json` (model B).

## LAC, α = 0.1 (target coverage 0.90)

Lower-tail coverage = 10th percentile of a center's coverage over the 100 splits.

**Model A**

| Method | Worst-center lower-tail | Lower-tail by center 0–5 | Lowest center mean coverage | Set size |
|---|---|---|---|---|
| Global | 0.786 | 0.858 0.967 0.917 **0.786** 0.830 0.942 | 0.819 | 2.45 |
| Client-wise | 0.794 | 0.874 0.875 0.860 0.858 0.843 **0.794** | 0.900 | 2.37 |
| Sample-weighted | 0.738 | 0.812 0.957 0.889 **0.738** 0.766 0.914 | 0.770 | 2.14 |
| FCP | 0.788 | 0.862 0.967 0.919 **0.788** 0.834 0.943 | 0.822 | 2.46 |
| ShrinkFCP | 0.819 | 0.869 0.911 0.892 **0.819** 0.836 0.936 | 0.856 | 2.36 |

**Model B**

| Method | Worst-center lower-tail | Lower-tail by center 0–5 | Lowest center mean coverage | Set size |
|---|---|---|---|---|
| Global | 0.670 | 0.857 0.975 0.926 0.852 **0.670** 0.782 | 0.717 | 2.39 |
| Client-wise | 0.795 | 0.875 0.873 0.860 0.857 0.844 **0.795** | 0.898 | 2.55 |
| Sample-weighted | 0.605 | 0.815 0.965 0.900 0.816 **0.605** 0.705 | 0.654 | 2.09 |
| FCP | 0.673 | 0.860 0.976 0.927 0.854 **0.673** 0.787 | 0.721 | 2.41 |
| ShrinkFCP | 0.722 | 0.870 0.909 0.895 0.854 **0.722** 0.786 | 0.771 | 2.32 |

APS and α = 0.05 / 0.2 show the same pattern (see JSON).

## RQ4 (pre-specified success criterion, LAC, α = 0.1)

| | Reference | Gain of ShrinkFCP | 95% CI | Set size ≤ 1.10 × ref | All centers' mean coverage ≥ 0.89 | Result |
|---|---|---|---|---|---|---|
| Model A | client-wise | +2.53 | [−2.82, +3.87] | yes | **no** (0.856, center 3) | **not met** |
| Model B | client-wise | −7.32 | [−14.01, −5.39] | yes | **no** (0.771, center 4) | **not met** |

k curve (worst-center lower-tail coverage): model A peaks at k = 50 (0.836; k = 0: 0.794, k = ∞: 0.786);
model B falls steadily from k = 0 (0.795) to k = ∞ (0.670). Data-driven k on validation lesions:
model A 10–100 (test 0.833), model B 0–10 (test 0.794).

**RQ4 is answered negatively**: ShrinkFCP does not meet the pre-specified criterion for either model.

## Reading

1. **Global federated conformal calibration fails on real centers (RQ1).** With one shared threshold
   (global, FCP, sample-weighted), some centers fall well short of the 90% target *on average*, not
   only in the tail: center 3 for model A (mean 0.82) and center 4 (MSK) for model B (mean 0.72), while
   other centers are over-covered (centers 1, 2, 5 at 0.92–0.98). FCP's finite-sample correction
   changes almost nothing. The cause is systematic differences between centers, not small samples.
2. **Client-wise calibration restores mean coverage in every center** (≈ 0.90), and its worst
   lower-tail coverage (0.79–0.80, in the smallest center 5) is close to the expected noise floor of
   about 0.82 for that center size.
3. **Why ShrinkFCP fails.** The protocol assumed that the small-center problem is variance, not a
   mean shortfall. On Fed-ISIC2019 the larger problem is between-center shift: the global threshold is
   biased for some centers. Shrinking toward it trades a little variance for that bias. For model A
   the trade is roughly neutral overall (worst-center tail +2.5 points, but center 3's mean coverage
   drops to 0.86); for model B, whose global threshold is badly biased at center 4, it is harmful
   (−7.3 points). The k curves show the same: the best k is small (A) or zero (B).
4. **Practical conclusion:** in federated dermoscopy with heterogeneous centers, conformal thresholds
   should be calibrated per center; pooling thresholds across centers, including shrinkage toward a
   pooled threshold, can leave a center far below its nominal coverage.
5. ShrinkFCP is not modified after these results. Any variant designed now (for example, shrinking
   only when a center's scores look exchangeable with the pool) would be exploratory and is reported
   as such if pursued.
