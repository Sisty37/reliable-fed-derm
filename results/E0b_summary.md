# E0b: the mother paper's data pipelines (protocol v0.10)

Run 30 Oct 2026, 14:12–15:43 (Asia/Dhaka), commit e3c939e (tag `e0b-preregistered`, pushed before
training). Model A, FedAvg, 3 Dirichlet clients, 9 rounds x 100 updates; choices in
`results/E0b_implementation_notes.md`. Metric: balanced accuracy (7 classes) on each arm's own test set.

## Leakage of each pipeline (property of the split, no training involved)

| Arm | Test rows whose lesion is in training | Test rows whose source image is in training |
|---|---|---|
| a, text (lesion split, train-only balancing) | 0% | 0% |
| b, tables (balance all 10,015 images to 22,520 rows, image split 70/15/15) | 86.8–87.1% | 82.4–83.8% |
| c, image split with train-only balancing | 28.3–28.8% | 0% |

In arm b, about 83% of the test rows are flipped or rotated copies of images used in training (or
those images themselves).

## Balanced accuracy

| Arm | Seed 0 | Seed 1 | Seed 2 | Mean (SD) |
|---|---|---|---|---|
| a | 0.6627 | 0.6467 | 0.6110 | 0.640 (0.027) |
| b | 0.6716 | 0.6826 | 0.6710 | 0.675 (0.007) |
| c | 0.6333 | 0.6439 | 0.5999 | 0.626 (0.023) |

Same-seed differences: b − a = +0.9, +3.6, +6.0 points (mean +3.5); c − a = −2.9, −0.3, −1.1 points
(mean −1.4).

Exploratory (not part of the E0b claim): on the test rows whose lesion is not in training, arm b
reaches 0.555, 0.353, 0.652 (436–445 rows, with few rows in the rare classes, hence the spread) and
arm c 0.624, 0.665, 0.618.

## Reading

1. The pipeline implied by the mother paper's tables puts copies of training images into about 83% of
   the test set. This is a property of the pipeline and does not depend on the model.
2. With our model A training (FedAvg, 9 rounds), this pipeline raises balanced accuracy by 3.5
   points on average over the lesion-grouped text pipeline (all three seeds positive, 3 seeds only,
   so no test is claimed).
3. Image-level splitting with training-only balancing (arm c, 28% lesion overlap) does not raise
   balanced accuracy here (−1.4 points, within seed variation).
4. The size of the inflation depends on how strongly a model memorizes its training images. Our
   model A reaches about 0.64–0.68, far below the mother paper's reported values, so these numbers
   show the direction and a lower bound for this training budget; they do not reproduce or explain
   the mother paper's values. FedHybridXAI itself is not re-run (no code link in the paper).
5. The three arms have different test sets (different size and class make-up; arm b's test set
   contains copies), so the between-arm differences mix leakage with test-set composition. The
   within-model comparison that separates the two is E1's primary outcome.
