# Reliable Federated Dermoscopy

Code and locked protocol for:
**Revisiting Federated Skin Lesion Benchmarks: Lesion-Level Leakage, Calibration and Conformal Coverage on Real Centers**

| Path | Content |
| --- | --- |
| `PROTOCOL.md` | Pre-specified design (locked at tag `v0.1-protocol`) |
| `scripts/audit_fedisic_leakage.py` | Lesion-leakage audit of the FLamby Fed-ISIC2019 split; writes size-matched splits |
| `scripts/sim_leakage_conformal.py` | Synthetic mechanism simulation (leakage vs conformal coverage) |
| `scripts/systematic_search.py` | Literature search on OpenAlex, PubMed and arXiv |
| `scripts/hash_splits.py` | Checksums for split files |
| `splits/` | Locked split files (added from the audit output) |
| `tests/` | Split integrity tests |

Quick start (Windows PowerShell):

    pip install -r requirements.txt
    git clone https://github.com/owkin/FLamby.git
    python scripts/audit_fedisic_leakage.py FLamby/flamby/datasets/fed_isic2019 --isic-meta ISIC_2019_Training_Metadata.csv --out splits
    python scripts/hash_splits.py
    python tests/test_splits.py
