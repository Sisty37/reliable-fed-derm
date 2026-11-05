# E6 part 2: PAD-UFES-20 domain-shift stress test, results (6 Nov 2026)

Code and choices committed and tagged `e6-padufes-prerun` (commit 000a7aa) before any prediction.
Run: `prepare_padufes.py`, `padufes_predict.py`, `padufes_analysis.py`. Output: `results/E6_padufes.json`,
`runs/padufes20/*/preds.npz`. Lesion-grouped FedAvg models, seeds 0–4; six shared classes.

## Data note (found during preprocessing)

`prepare_padufes.py` reported 250 `lesion_id` values used by two patients. These are reused numbers,
not shared lesions: in 226 of the 250 the two patients' diagnoses differ, and image names
(`PAT_<patient>_<lesion>_<n>`) always match the metadata. The lesion key is therefore
(patient_id, lesion_id): 1,891 lesions (the dataset's "1,641" counts distinct lesion numbers).
All PAD-UFES-20 analyses split and resample by **patient**, so no result depends on this; the check
in `prepare_padufes.py` assumed globally unique lesion IDs and its "must be 0" message was wrong.

## Accuracy and calibration (6 classes; PAD-UFES-20 interval = patient bootstrap)

| | Model A | Model B |
|---|---|---|
| BA, Fed-ISIC2019 lesion-grouped test | 0.471 | 0.522 |
| BA, PAD-UFES-20 | 0.356 [0.333, 0.380] | 0.342 [0.320, 0.366] |
| ECE, Fed-ISIC2019 → PAD-UFES-20 | 0.058 → 0.190 [0.175, 0.207] | 0.034 → 0.212 [0.194, 0.231] |
| Brier, Fed-ISIC2019 → PAD-UFES-20 | 0.475 → 0.865 | 0.490 → 0.830 |
| PAD recall MEL / NV / BCC / AK / BKL / SCC | 0.22 / 0.73 / 0.33 / 0.08 / 0.57 / 0.20 | 0.12 / 0.42 / 0.42 / 0.19 / 0.73 / 0.17 |

The foundation-model encoder loses more balanced accuracy (−18.0 vs −11.5 points) and ends no
better on smartphone images; ECE triples (A) to sextuples (B). AK, the largest PAD-UFES-20 class
after BCC, is almost never recognized.

## Conformal prediction (LAC primary)

| Score, α | Transfer: Fed-ISIC cov → PAD cov [95% CI], set size | Recalibrated on PAD patients: cov, q10, set size |
|---|---|---|
| A LAC 0.1 | 0.900 → **0.596 [0.576, 0.614]**, 2.73 | 0.900, 0.880, **4.87** |
| B LAC 0.1 | 0.900 → **0.669 [0.650, 0.687]**, 2.44 | 0.900, 0.884, **4.30** |
| A LAC 0.05 / 0.2 | 0.738 / 0.407 | 0.950 / 0.800 (size 5.24 / 4.17) |
| B LAC 0.05 / 0.2 | 0.793 / 0.519 | 0.949 / 0.801 (size 4.89 / 3.30) |
| A APS 0.1 | 0.715 [0.698, 0.731], 3.68 | 0.902, 0.885, 4.79 |
| B APS 0.1 | 0.752 [0.735, 0.767], 3.11 | 0.900, 0.881, 4.37 |

Per-class coverage (A / B, LAC 0.1, transfer): AK 0.35 / 0.48, SCC 0.46 / 0.46, BCC 0.63 / 0.79.

## Reading

1. **Calibration transfer fails under modality shift** (pre-specified rule, item 7 of the notes):
   coverage 0.60 (A) and 0.67 (B) against a target of 0.90, intervals far below 0.90, for every α and
   both scores. The guarantee holds on Fed-ISIC2019 (0.900) and does not carry over.
2. **Local recalibration restores coverage but not usefulness:** with half of the PAD-UFES-20
   patients as calibration set, coverage returns to 0.90 (10th percentile 0.88), but sets contain
   4.3–4.9 of 6 classes on average, i.e. the model is close to uninformative on this domain.
   Coverage is restored by enlarging sets, not by better predictions.
3. **The coverage gap concentrates on the classes the model confuses** (AK, SCC, BCC), and those
   stay below 0.90 even after recalibration (AK 0.82 / 0.80, SCC 0.84 / 0.80), because marginal
   recalibration does not give class-conditional coverage.
4. **No robustness advantage for the foundation model** in accuracy; its transfer coverage is
   somewhat higher (0.67 vs 0.60) with smaller sets, but both fail. PAD-UFES-20 is not listed among
   PanDerm's pretraining sources (not proof of not seen).
5. APS at α = 0.05 over-covers after recalibration (0.98 / 0.96) because nearly full sets make the
   non-randomized score discrete; this is expected and not a failure.
