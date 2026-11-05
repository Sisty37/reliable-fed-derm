# Exploratory nearest-neighbour ("proximity") check: choices fixed before running (6 Nov 2026)

**Status: exploratory**, not part of the pre-registered protocol; reported as such.

Question: frozen encoders (A-frozen, B) cannot memorize training images, yet their linear heads show
leakage inflation of about 8 points (E1). Is this because a leaked test image has a sibling image of
the same lesion in training that lies very close to it in embedding space (proximity), so the head's
decision regions there are set by that sibling?

Code: `scripts/nn_proximity.py`. Data: plain-view embeddings (`embeddings/effb0.npz`,
`embeddings/panderm.npz`), centred with the training mean and L2-normalized; image-level E1 splits with
an existing FedAvg linear head (all seeds found on disk, at least 0–2).

Reported (mean and sd over seeds, no hypothesis test):
1. cosine similarity of each test image to its nearest training image, leaked vs new-lesion test images;
   share of leaked test images whose nearest neighbour is a sibling image of the same lesion;
2. 1-NN classifier balanced accuracy on full, leaked and new-lesion test images, and its inflation;
3. FedAvg linear-head accuracy on leaked images with a sibling nearest neighbour vs leaked images
   without one vs new-lesion images, and the head's agreement with the 1-NN label.

Expected pattern if proximity explains the inflation: leaked images are much closer to training data
than new ones, their nearest neighbour is usually a sibling, the 1-NN classifier shows a large
inflation, and the head is most accurate on leaked images with a sibling nearest neighbour.
Balanced accuracy is unweighted over the classes present (descriptive; not the Gate 2 measure).
