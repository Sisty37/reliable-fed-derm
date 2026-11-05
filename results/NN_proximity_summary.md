# Exploratory nearest-neighbour ("proximity") check: results (6 Nov 2026)

**Exploratory** (not in the protocol). Code and notes committed and tagged `nn-prerun` (commit 3a1051e)
before running `scripts/nn_proximity.py`. Output: `results/NN_proximity.json`. Image-level E1 splits;
A-frozen seeds 0–2, model B seeds 0–4; mean over seeds.

| | A-frozen (EfficientNet-B0) | B (PanDerm) |
|---|---|---|
| Leaked share of test images | 0.595 | 0.595 |
| Nearest-train cosine similarity, leaked / new | 0.807 / 0.748 | 0.852 / 0.793 |
| Leaked images whose nearest neighbour is a sibling | 0.539 | 0.581 |
| 1-NN BA, full / leaked / new | 0.594 / 0.694 / 0.367 | 0.622 / 0.716 / 0.398 |
| **1-NN inflation (full − new)** | **+22.7 (sd 1.1)** | **+22.4 (sd 1.4)** |
| **Linear head inflation (full − new)** | **+6.2 (sd 1.6)** | **+6.6 (sd 3.7)** |
| Head accuracy, leaked with sibling NN / leaked other NN | 0.568 / 0.515 | 0.626 / 0.588 |
| Head agrees with 1-NN label, leaked / new | 0.519 / 0.609 | 0.568 / 0.629 |

(Balanced accuracy unweighted over classes, no center reweighting; the head inflation here, +6.2 and
+6.6, is close to but not the same measure as the Gate 2 values, +8.2 and +8.7. Raw accuracies of
leaked vs new images are confounded by class mix and are not compared directly.)

## Reading

1. **Proximity is real for frozen encoders.** Leaked test images lie closer to the training set than
   new-lesion images (cosine +0.06 for both encoders), and for more than half of them the single
   nearest training image is a sibling image of the same lesion.
2. **A classifier that fully exploits proximity inflates by about 22 points.** The 1-NN classifier,
   the extreme of memorization, gains +22.7 and +22.4 points from leakage with either encoder.
3. **The linear heads realize only a small part of it (about 6 points)**, and benefit modestly when
   the nearest neighbour is a sibling (+5.3 and +3.8 points accuracy over leaked images without one).
   The head does not simply copy the nearest neighbour (agreement 0.52–0.57 on leaked images).
4. **Link to E7 (interpretation):** across all models the inflation grows with the classifier's
   ability to exploit nearby training images: linear head (~6–9) < FedPer-style 256-unit head (B, 10.6
   in centers 0/2/3) < full fine-tuning with 20 epochs (A pooled/local, 12–14) < 1-NN (~22). Leakage
   creates a fixed potential (siblings close to test images); how much of it shows up as inflated
   accuracy depends on capacity. Frozen encoders do not remove leakage bias; they reduce how much of
   it a low-capacity head can use.
