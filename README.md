# Reliable Federated Dermoscopy

Code, data splits, predictions and protocol history for:
**Lesion-level leakage and site-level reliability in Fed-ISIC2019: an audit of federated skin lesion classification**

All commands run from the repository folder (Windows PowerShell shown; Linux/macOS work the same with `/`).

## Environment

    pip install -r requirements.txt          # exact versions: requirements-lock.txt
    python scripts\record_environment.py      # writes results\Environment.json (Python, GPU, SHA-256 checksums)

Data: Fed-ISIC2019 via FLamby (commit edacf54) in `data\fedisic`, PAD-UFES-20 in `data\padufes20`,
PanDerm-Base checkpoint in `..\models\panderm` (not redistributed).

## Workflow

| Step | Command | Output |
| --- | --- | --- |
| Splits | `python scripts\prepare_fedisic.py`, `scripts\make_e1_splits.py`, `scripts\make_e8_splits.py`, `scripts\hash_splits.py` | `splits\` |
| E0, E1, E2, E7 (model A) | `python scripts\train_fl.py --method <m> --seed <s> --split <csv>`; `scripts\finetune_fedavg.py`; `scripts\run_e7.ps1` | `runs\` |
| Frozen encoders | `python scripts\extract_embeddings.py`; `python scripts\train_linear.py --encoder {effb0,panderm} ...` | `embeddings\`, `runs\` |
| E6 (LOCO, PAD-UFES-20) | `train_fl.py` / `train_linear.py` with `--exclude-center`; `scripts\padufes_predict.py` | `runs\` |
| E8, E9 | `scripts\run_e8_modelA.ps1`, `scripts\run_e8e9_linear.ps1` | `runs\sibexp_*`, `runs\resplit_*` |
| Budget controls (post hoc) | `scripts\run_budget_ft.ps1` | `runs\` |
| Gate 2, E2/E7, E3/E6, E5/E6 conformal, E8 | `gate2.py`, `e2_analysis.py`, `e3_e6_analysis.py`, `e5_conformal.py [--loco]`, `padufes_analysis.py`, `e8_analysis.py` | `results\*.json` |
| Post hoc analyses | `scripts\run_posthoc_analyses.ps1` (audit, conformal and calibration sensitivity, class-conditional, EB variants, simulation), `nn_matched.py`, `budget_analysis.py`, `followup_analyses.py` | `results\*.json` |
| Tables, figures, supplement | `make_tables.py`, `make_figures.py`, `make_supplement.py` | `tables\`, `figures\`, `paper\supplement.tex` |
| Check | `python scripts\check_runs.py`; `python scripts\verify_numbers.py` | consistency report |

## Main tables and figures

| Item | Input results | Script |
| --- | --- | --- |
| Table 1 | `Audit_supplement.json`, `E7_analysis.json` | `make_tables.py` |
| Table 2 | `E1_gate2.json`, `E7_analysis*.json`, `E5/E6_conformal*.json`, `E6_padufes.json`, `E8_sibling_exposure.json`, `Conformal_sensitivity.json` | `make_tables.py` |
| Table 3 | `E7_analysis*.json` | `make_tables.py` |
| Table 4 | `E6_conformal_loco*.json`, `Conformal_sensitivity.json` | `make_tables.py` |
| Fig. 1 | `splits\split_official.csv`, images | `make_figures.py` |
| Fig. 3 | `E1_gate2*.json` | `make_figures.py` |
| Fig. 4, 5 | `E7_analysis*.json`, `NN_proximity.json` | `make_figures.py` |
| Fig. 6 | `E3_E6_analysis*.json`, `E7_calibration*.json` | `make_figures.py` |
| Fig. 7 | `E6_conformal_loco*.json`, `Conformal_sensitivity.json` | `make_figures.py` |
| Fig. 8 | `E6_padufes.json`, `E3_E6_analysis*.json` | `make_figures.py` |
| Supplement S1–S18 | all `results\*.json` | `make_supplement.py` |

## Protocol

`PROTOCOL.md` holds the protocol and its amendment log; git tags `v0.1-protocol` to `v0.11-protocol` and the
pre-run tags (`*-preregistered`, `*-prerun`) mark the states before each batch of runs. Supplementary S8 maps
versions to commits and tags.
