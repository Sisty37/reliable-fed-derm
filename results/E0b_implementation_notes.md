# E0b: implementation choices fixed before any E0b training (30 Oct 2026)

Protocol v0.10 specifies the three arms of E0b. The points below are not stated in the protocol and
were fixed, with the code (`scripts/make_e0b_splits.py`, `scripts/train_e0b.py`) and the split files
(`splits/e0b/`), in a commit made **before** the first E0b training run. The split files were built
with seeds 0, 1, 2 and are not edited afterwards.

1. **Classes.** ISIC 2019 labels HAM10000's akiec images as AK (130) or SCC (197, Bowen's disease).
   The mother paper uses the 7 HAM10000 classes, so the two are merged back into one akiec class
   (327 images, as in the mother paper's Table 3). Model A gets a 7-class head.
2. **Loss.** Focal loss with gamma 2 and **no class weights**, because each arm's training set is
   already balanced by the pipeline; FLamby's class weights would re-weight a balanced set.
3. **Training.** As in E0: EfficientNet-B0 (ImageNet), FLamby augmentations, FedAvg, 9 rounds x 100
   local updates, SGD lr 0.01, batch 64, AMP, final model = client 0 copy, no checkpoint selection
   (validation folds are built but not used).
4. **Balancing details.** Targets T_c from Table 3 (sum 22,520). Classes above target (NV) are
   undersampled without replacement. Classes below target get copies; sources are taken in a new
   random order on each pass, and the copy made on pass k uses transform ((k - 1) mod 5) + 1
   (h-flip, v-flip, rot90, rot180, rot270), so a source is copied with a repeated transform only after
   5 passes. In arms a and c the same absolute targets are applied to the training set only.
5. **Clients.** Dirichlet(alpha = 0.5) class partition of the training rows (copies included) over 3
   clients; a draw is repeated if any client gets less than 5% of the rows.
6. **Seed.** One seed per run controls the split, the balancing, the client partition and training.
7. **Exploratory output.** Balanced accuracy on the test rows whose lesion is not in training is saved
   for every arm but is not part of the E0b claim.

## Split properties (from `splits/e0b/e0b_split_summary.csv`, no training involved)

| Arm | Train / val / test rows | Test rows whose lesion is in training | Test rows whose source image is in training |
|---|---|---|---|
| a (text: lesion split, train-only balancing) | 22,520 / ~2,000 / ~2,025 | 0% | 0% |
| b (tables: balance all, image split 70/15/15) | 15,764 / 3,378 / 3,378 | 86.8–87.1% | 82.4–83.8% |
| c (image split, train-only balancing) | 22,520 / 2,003 / 2,003 | 28.3–28.8% | 0% |

Ranges are over seeds 0, 1, 2.
