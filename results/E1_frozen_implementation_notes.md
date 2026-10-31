# E1, models A-frozen and B: implementation choices fixed before any frozen-encoder run (31 Oct 2026)

Code: `scripts/extract_embeddings.py`, `scripts/train_linear.py`, `scripts/gate2.py --prefix`.
Protocol v0.10, Section 4 sets the design (frozen encoder, 5 augmented training views, plain view for
validation and test, federated linear head, convergence check). Points the protocol leaves open:

1. **Views.** Each image gets 1 plain view (FLamby test transform: CenterCrop 200, ImageNet
   normalization) and 5 augmented views (FLamby training transforms). View v of image i uses random
   seed 1,000,003 × v + i, and FLamby's once-drawn CoarseDropout count is fixed by seeding Python's
   `random` with 20261031 before the transform is built. Both encoders get the same pixels.
2. **Model A-frozen.** EfficientNet-B0 with ImageNet weights (same as model A), eval mode,
   global-average-pooled 1280-d features, 200 × 200 input as in model A.
3. **Model B.** PanDerm-Base (ViT-B/16, `panderm_base_patch16_224` from the PanDerm repository,
   checkpoint `panderm_bb_data6_checkpoint-499.pth`), eval mode, CLS token after the final LayerNorm
   (PanDerm's own linear-probe setting), 768-d. The 200 × 200 view is resized to 224 × 224 (bilinear)
   because the model's position embeddings are fixed at 14 × 14 patches. ImageNet normalization, as in
   FLamby (PanDerm's code uses the same mean, with std 0.228 instead of 0.229 for the red channel).
   Weights must load with no missing non-head key, otherwise extraction stops.
4. **Inference** in fp16 autocast; embeddings stored in fp16.
5. **Linear heads.** One augmented view, drawn at random, each time a training row is used. FedAvg as
   model A (SGD lr 0.01, batch 64, 9 rounds, FLamby weighted focal loss, client 0 copy evaluated) with
   U local updates per round from the convergence check. Pooled: Adam lr 5e-4, batch 64, 20 epochs
   (protocol's pooled setting), final head kept.
6. **Convergence check** (once per encoder: lesion-grouped split, seed 0, FedAvg). "Pooled training
   loss" is the loss of the client 0 head on all training rows, averaged over the 5 views, after each
   round. Starting at U = 100, U is doubled while that loss falls by more than 1% from round 8 to
   round 9. The chosen U is saved in `results/linear_updates_<encoder>.json` and used for every split,
   seed and method of that encoder.
7. **Model B status.** Pretraining overlap not ruled out (`results/PanDerm_overlap_check.md`).
