# E6 (leave-one-center-out): implementation choices fixed before any E6 run (3 Nov 2026)

Protocol v0.11, Section 6 (E6) and Section 5b. Code: `scripts/train_fl.py --exclude-center h` (model A)
and `scripts/train_linear.py --encoder panderm --exclude-center h` (model B).

1. **Models.** FedAvg, 9 rounds, lesion-grouped splits `splits/e1/resplit_lesion_seed{0,1,2}_val.csv`;
   for each held-out center h = 0..5, the federation has the other 5 clients and center h's training
   data is never used (model B: also not in the federated centring statistics). 6 × 3 = 18 runs per
   model; run names `fedavg_r9_loco{h}_s{seed}` and `pandermfrozen_fedavg_r9_loco{h}_s{seed}`.
2. **Evaluation.** The client copy of the first training client (FLamby convention) is evaluated on
   the test and validation lesions of all 6 centers, including the held-out one.
3. **Settings** are those of E1 (model A: 100 local updates per round; model B: centred embeddings,
   U = 6400). No setting is tuned for E6.
4. **Use.** RQ1 (accuracy, calibration and coverage on an unseen center) and the Section 5b test of
   EB-ShrinkFCP and ShrinkFCP. The PAD-UFES-20 stress test follows as a separate step.
