# E1, model A, 9 rounds: Gate 2 (protocol v0.10)

Run 30 Oct 2026, 19:19 to 23:52 (Asia/Dhaka), commit 3e2509f (tag `e1-preregistered`, pushed before
training). Splits `splits/e1/resplit_{image,lesion}_seed{0,1,2}_val.csv` (locked re-splits plus a 10%
lesion-grouped validation fold). Analysis `scripts/gate2.py`, 2,000 lesion-bootstrap replicates; full
output in `results/E1_gate2.json`.

## Primary outcome: inflation of model A FedAvg (9 rounds)

| Seed | BA, full image-level test set | BA, new-lesion subset (reweighted) | Inflation | New-lesion test images |
|---|---|---|---|---|
| 0 | 0.5676 | 0.4816 | +8.60 | 1,858 / 4,648 |
| 1 | 0.5729 | 0.4398 | +13.31 | 1,950 / 4,648 |
| 2 | 0.5914 | 0.5670 | +2.43 | 1,845 / 4,648 |
| **Mean** | | | **+8.11 points, 95% CI [+4.74, +11.18]** | |

Seed range +2.43 to +13.31 points (the interval reflects test-set sampling only; Section 9).

**Gate 2: PASS** (inflation ≥ 2 points, interval above zero). RQ2 is the headline.

## Secondary (no decision attached)

| | FedAvg, 9 rounds | Pooled, 20 epochs |
|---|---|---|
| Inflation (reweighted new-lesion subset) | +8.11 [+4.74, +11.18] | +16.16 [+12.50, +19.42] |
| Between-condition difference (sensitivity) | 0.5773 − 0.5086 = +6.87 | 0.7274 − 0.5756 = +15.18 |
| Melanoma recall, full vs new-lesion (seeds 0/1/2) | 0.480/0.370, 0.494/0.355, 0.538/0.406 | 0.740/0.630, 0.609/0.452, 0.651/0.486 |

- The sensitivity check points the same way as the primary measure for both training regimes.
- Pooled training shows about twice the inflation of FedAvg. Pooled training sees every image more
  often, which fits the protocol's expectation that stronger fitting means more memorization; the
  27-round FedAvg runs (pending) test this within FedAvg.
- Limitation (Section 3): new-lesion images are mostly single-image lesions, which may differ from
  multi-image lesions in difficulty even after center reweighting. The agreement with the
  between-condition difference, which does not rely on this subset, partly addresses it.
