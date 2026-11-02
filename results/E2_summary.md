# E2, model A: method rankings, H3b (interim) and per-center leakage effects

Runs 1–3 Nov 2026 (seeds 3–4 and the new methods at commit 0c30675; FedAvg and pooled seeds 0–2 reused
from E1). 7 methods × 5 seeds × 2 split types = 70 runs. Analysis `scripts/e2_analysis.py` (commit
4bed4e4, choices in `results/E2_implementation_notes.md`), 2,000 lesion-bootstrap replicates; full
output in `results/E2_analysis.json`.

## Rankings (balanced accuracy, mean of 5 seeds)

| Method | Pooled BA, image-level | Pooled BA, lesion-grouped | Mean-of-centers BA, image-level | Mean-of-centers BA, lesion-grouped |
|---|---|---|---|---|
| Pooled | 0.7331 (1) | 0.5659 (1) | 0.6839 (1) | 0.6075 (1) |
| Local-only | 0.7295 (2) | 0.5330 (2) | 0.6328 (2) | 0.5088 (6) |
| SCAFFOLD | 0.6256 (3) | 0.5289 (3) | 0.5558 (5) | 0.5357 (4) |
| FedBN | 0.6075 (4) | 0.5136 (4) | 0.5927 (3) | 0.5676 (2) |
| FedProx | 0.5822 (5) | 0.4875 (7) | 0.5252 (7) | 0.5082 (7) |
| FedAvg-FT | 0.5792 (6) | 0.5012 (5) | 0.5619 (4) | 0.5563 (3) |
| FedAvg | 0.5770 (7) | 0.4947 (6) | 0.5355 (6) | 0.5226 (5) |

| Kendall's τ | Pooled BA (primary) | Mean-of-centers BA (secondary) |
|---|---|---|
| Between split types, seed-mean rankings | +0.81 [+0.62, +1.00] | +0.62 [+0.33, +0.81] |
| Between split types, single-seed rankings | +0.65 [+0.44, +0.69] | +0.27 [+0.14, +0.46] |
| Noise floor: different seeds, same split type | +0.70 [+0.51, +0.71] | +0.57 [+0.31, +0.58] |

## H3b (interim, without FedPer): inflation in centers 0, 2, 3, weighted by new-lesion counts

| FedAvg | FedProx | SCAFFOLD | FedBN | FedAvg-FT | Pooled | Local-only |
|---|---|---|---|---|---|---|
| +4.29 | +4.38 | +3.91 | +4.92 | +4.79 | +12.06 | +13.72 |

Personalized/local group (local-only, FedAvg-FT, FedBN) +7.81 minus FedAvg +4.29 = **+3.52 points,
95% CI [+2.26, +4.75]: supported (interim)**.

## Per-center inflation (points, seed mean) and dose-response

| Center | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Leakage rate (image-level test) | 83.0% | 0.2% | 68.0% | 43.5% | 2.7% | 50.2% |
| New-lesion test images | 421 | 789 | 214 | 255 | 160 | 44 |
| FedAvg | +10.0 | 0.0 | −2.5 | +0.6 | +0.2 | +1.9 |
| FedProx | +9.5 | 0.0 | −1.2 | +0.7 | −0.2 | +5.2 |
| SCAFFOLD | +9.8 | 0.0 | 0.0 | −2.6 | +0.3 | −4.6 |
| FedBN | +11.3 | 0.0 | −1.3 | −0.3 | +0.1 | −0.7 |
| FedAvg-FT | +11.0 | 0.0 | −2.6 | +0.8 | −0.1 | +0.4 |
| Pooled | +22.2 | 0.0 | +1.4 | +4.3 | +0.5 | +5.7 |
| Local-only | +21.7 | 0.0 | +4.7 | +8.2 | +0.3 | +3.9 |

Weighted least-squares slope of per-center inflation on leakage rate (points per 100% leakage):
FedAvg +8.1 [+5.0, +12.2], FedProx +8.3 [+5.2, +12.5], SCAFFOLD +8.2 [+5.3, +12.0], FedBN +9.5 [+6.5, +13.4],
FedAvg-FT +9.0 [+5.9, +12.8], pooled +20.6 [+17.6, +24.0], local-only +21.5 [+18.5, +25.1].

## Reading (secondary, no decision attached)

1. **Rankings on the primary metric barely change.** With pooled balanced accuracy, the seed-mean
   rankings of the two split types agree closely (τ = 0.81), and the single-seed agreement across split
   types (0.65) is within the seed-to-seed noise floor (0.70). Leakage lowers every method's score but
   leaves their order on the FLamby metric about as stable as re-running with another seed. This is
   reported as a small or null effect on rankings.
2. **One method changes place on the per-center metric.** With mean-of-centers balanced accuracy,
   local-only drops from 2nd (image-level) to 6th (lesion-grouped), and the single-seed agreement
   across split types (0.27) falls below the noise floor (0.57), with overlapping intervals. Image-level
   splits make local training look competitive with federated learning; lesion-grouped splits do not.
3. **H3b is supported (interim), but the effect comes from local-only.** Local-only (+13.7) and pooled
   (+12.1) gain most from leakage. FedBN (+4.9) and FedAvg-FT (+4.8) are only slightly above FedAvg
   (+4.3). The data support "the more a model fits the leaked center's own images, the more it gains",
   rather than a general advantage of personalized federated methods. Final test after FedPer (E7).
4. **Leakage effects sit in center 0.** Center 0 (BCN, 83% of test images leaked) carries most of the
   inflation (+10 to +22 points); centers 2 and 3 show small or no effects despite 68% and 44% leakage,
   and centers 1 and 4 have almost no leakage. All dose-response slopes are positive, and those of
   pooled and local-only are about twice those of the federated methods.
5. **Caveats.** Centers 2–5 have few new-lesion images (44–255), so their per-center values are
   noisy. The bootstrap intervals of the noise floor sit mostly below its point estimate (resampling
   adds noise to rankings), so the τ comparison is read from the point estimates and the overlap of
   intervals, not as a test.
