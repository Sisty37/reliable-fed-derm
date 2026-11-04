# Protocol 5b test: EB-ShrinkFCP on the E6 leave-one-center-out models (pre-specified, v0.11)

Run 4 Nov 2026 with `scripts/e5_conformal.py --loco` (code committed at e36d6a6, tag `e6-runs-done`,
before the test was run; protocol tag `v0.11-protocol` before any E6 model was trained). 18 models per
encoder (held-out center 0–5 × seeds 0–2), 100 lesion-wise calibration/evaluation splits each,
2,000 lesion-bootstrap replicates. Output: `results/E6_conformal_loco.json`,
`results/E6_conformal_loco_pandermfrozen.json`.

## LAC, α = 0.1, E6 models (worst-center lower-tail coverage; lowest center mean coverage; set size)

| Method | Model A | | | Model B | | |
|---|---|---|---|---|---|---|
| Global | 0.783 | 0.829 | 2.57 | 0.681 | 0.751 | 2.57 |
| Client-wise (reference) | 0.814 | 0.900 | 2.41 | 0.798 | 0.900 | 2.60 |
| Sample-weighted | 0.733 | 0.792 | 2.27 | 0.626 | 0.694 | 2.22 |
| FCP | 0.785 | 0.831 | 2.59 | 0.684 | 0.755 | 2.59 |
| ShrinkFCP | 0.815 | 0.860 | 2.45 | 0.731 | 0.793 | 2.43 |
| **EB-ShrinkFCP** | **0.837** | **0.899** | 2.50 | **0.803** | **0.898** | 2.61 |

## Pre-specified tests (reference = client-wise for both models)

| | Gain | 95% CI | Set size ≤ 1.10 × ref | All centers' mean coverage ≥ 0.89 | Result |
|---|---|---|---|---|---|
| EB-ShrinkFCP, model A | +2.28 | [+0.78, +3.88] | yes | yes | **met** |
| EB-ShrinkFCP, model B | +0.55 | [−1.70, +0.94] | yes | yes | not met (gain < 2, CI includes 0) |
| ShrinkFCP, model A | +0.09 | [−2.12, +1.98] | yes | no | not met |
| ShrinkFCP, model B | −6.67 | [−11.59, −5.13] | yes | no | not met |

The protocol requires success **for each of models A and B**, so the Section 5b criterion is **not
met overall**: it is met for model A and not for model B.

## Post hoc (E5 models, seeds 0–4; method specified after these data were seen)

EB-ShrinkFCP vs client-wise: model A +4.83 [+0.20, +6.14], all guards met; model B +0.06 [−1.97, +2.20].
Re-running E5 reproduced the earlier ShrinkFCP results exactly (model A +2.53 [−2.82, +3.87];
model B −7.32 [−14.01, −5.39]).

## Reading

1. **EB-ShrinkFCP is safe where ShrinkFCP is not.** In all four settings (E5/E6 × models A/B) it keeps
   every center's mean coverage at about 0.90 (0.897–0.901 with LAC), while ShrinkFCP drops the worst center to
   0.77–0.86. Its set sizes stay within 4% of client-wise calibration.
2. **It improves the tail where there is room.** For model A it raises the worst-center lower-tail
   coverage by 2.3 points on held-out models (pre-specified, interval above zero) and by 4.8 points on
   the E5 models (post hoc). For model B it matches client-wise calibration (+0.6, +0.1) but does not
   improve it.
3. **Why.** The between-center variance estimate decides how much each center borrows. When the
   global threshold is strongly biased for some center (model B, center 4), τ² is large and
   EB-ShrinkFCP falls back to near client-wise thresholds instead of importing the bias, which is
   exactly where fixed-k ShrinkFCP fails (−6.7 points).
4. **What can be claimed:** a pre-specified, held-out improvement for one of the two models, and no
   harm for the other; not a general improvement. The noise floor of the smallest center (about 0.82)
   limits how much any calibration rule can raise the worst-center tail.
