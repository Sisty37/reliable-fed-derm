# E2, model B (PanDerm-Base, frozen, linear heads): rankings, H3b (interim), per-center effects

Runs 3 Nov 2026 (commit tagged `e2-modelB-preregistered`; FedAvg and pooled seeds 0–2 reused from E1).
6 methods × 5 seeds × 2 split types. Analysis `scripts/e2_analysis.py --prefix pandermfrozen_`
(FedBN does not apply), 2,000 lesion-bootstrap replicates; output `results/E2_analysis_panderm.json`.
Model B: pretraining overlap not ruled out.

## Rankings (balanced accuracy, mean of 5 seeds)

| Method | Pooled BA, image-level | Pooled BA, lesion-grouped | Mean-of-centers, image-level | Mean-of-centers, lesion-grouped |
|---|---|---|---|---|
| SCAFFOLD | 0.6176 (1) | 0.5419 (3) | 0.5578 (3) | 0.5548 (2) |
| Local-only | 0.6088 (2) | 0.5662 (1) | 0.5865 (2) | 0.5502 (4) |
| FedAvg-FT | 0.6021 (3) | 0.5292 (4) | 0.5441 (4) | 0.5536 (3) |
| Pooled | 0.5992 (4) | 0.5625 (2) | 0.5922 (1) | 0.5925 (1) |
| FedProx | 0.5911 (5) | 0.5229 (6) | 0.5323 (5) | 0.5451 (6) |
| FedAvg | 0.5904 (6) | 0.5232 (5) | 0.5322 (6) | 0.5459 (5) |

| Kendall's τ | Pooled BA (primary) | Mean-of-centers BA |
|---|---|---|
| Between split types, seed-mean rankings | +0.47 [+0.07, +0.73] | +0.60 [−0.07, +1.00] |
| Between split types, single-seed rankings | +0.49 [+0.08, +0.58] | +0.38 [+0.12, +0.54] |
| Noise floor (different seeds, same split type) | +0.69 [+0.35, +0.69] | +0.47 [+0.12, +0.55] |

## H3b (interim): inflation in centers 0, 2, 3, weighted by new-lesion counts

FedAvg +6.08, FedProx +5.85, SCAFFOLD +5.98, FedAvg-FT +6.27, pooled +2.67, local-only +4.15.
Personalized/local group (local-only, FedAvg-FT) +5.21 minus FedAvg +6.08 = **−0.87 points,
95% CI [−1.85, +0.24]: not supported (interim)**.

## Per-center inflation (points, seed mean)

| Center | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Leakage rate | 83.0% | 0.2% | 68.0% | 43.5% | 2.7% | 50.2% |
| FedAvg | +12.0 | 0.0 | +1.5 | +0.2 | 0.0 | +1.4 |
| FedProx | +11.4 | 0.0 | +1.6 | +0.2 | 0.0 | +1.4 |
| SCAFFOLD | +11.7 | 0.0 | +2.0 | −0.2 | −0.2 | +1.5 |
| FedAvg-FT | +12.1 | 0.0 | +1.8 | +0.3 | −0.1 | −0.1 |
| Pooled | +7.5 | 0.0 | −0.7 | −2.5 | +0.2 | +2.4 |
| Local-only | +8.7 | 0.0 | +0.6 | −0.3 | −0.1 | +1.9 |

Dose-response slopes (points per 100% leakage): FedAvg +11.1 [+8.0, +15.3], FedProx +10.7 [+7.5, +14.8],
SCAFFOLD +11.1 [+7.9, +14.9], FedAvg-FT +11.3 [+8.1, +15.4], pooled +6.2 [+2.8, +10.4],
local-only +7.9 [+4.7, +12.3].

## Reading (secondary; 5 seeds)

1. **Rankings are less stable for model B than for model A.** On the primary metric the single-seed
   agreement across split types (0.49) is below the noise floor (0.69), with overlapping intervals:
   SCAFFOLD moves from 1st (image-level) to 3rd, pooled from 4th to 2nd, local-only from 2nd to 1st.
   The six methods lie within 3 points of each other on image-level splits, so small shifts reorder
   them; this is read as fragile rankings, not as a clear leakage effect.
2. **H3b does not hold for the frozen encoder.** With a frozen encoder and a linear head, pooled and
   local-only training gain *less* from leakage than the federated methods (+2.7 and +4.2 vs about
   +6). Together with model A (local-only +13.7, pooled +12.1, FedAvg +4.3), this indicates that the
   extra inflation of local and pooled training needs a trainable encoder that can fit the leaked
   center's images; with a fixed encoder, every method shows a similar, smaller inflation.
3. **Center 0 again carries the effect** (+7.5 to +12.1 points); all dose-response slopes are
   positive.
4. **Interpretation of H3b overall** (to be finalized after FedPer, E7): supported for model A,
   driven by local-only and pooled; not supported for model B. The protocol tests H3b per model, so
   both results are reported.
