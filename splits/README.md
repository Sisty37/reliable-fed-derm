# Split files (locked)

Copy the files produced by `scripts/audit_fedisic_leakage.py` (folder `results/splits/` on your PC) here:

- `split_official.csv`, `split_clean.csv`, `split_control.csv`
- `resplit_image_seed0.csv` ... `resplit_image_seed4.csv`
- `resplit_lesion_seed0.csv` ... `resplit_lesion_seed4.csv`

Then run `python scripts/hash_splits.py` to write `splits/SHA256SUMS.txt`, and commit before tagging `v0.1-protocol`.
These files must never be edited by hand.
