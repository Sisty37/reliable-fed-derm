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

## Deviation, 31 Oct 2026 (after the first convergence check, before any test-set result)

**What happened.** The embeddings were extracted (10.6 min) and the convergence check was run on the
raw embeddings. For EfficientNet-B0 the pooled training loss fell smoothly (9.44 to 6.75 over 9
rounds; U = 100). For PanDerm-Base it jumped between rounds at every U tried (e.g. U = 100:
21.8, 30.4, 20.4, 18.4, 40.9, 19.5, 13.9, 15.7, 13.2), and the rule was met at U = 1600 only because two
noisy values happened to be close (8.08, 8.06). The training was unstable, not converged.

**Diagnosis** (plain-view embeddings, all images): per-dimension SD 0.019–0.929 and largest absolute
value 6.1 for EfficientNet-B0; per-dimension SD 0.000–0.887 but largest absolute value 16.3 for
PanDerm-Base, i.e. some PanDerm dimensions carry large, almost constant offsets (a known property of
ViT features). With SGD lr 0.01 and FLamby's class weights (up to 101), these offsets give very large
gradients on the matching head weights.

**Change.** Before the linear head, every embedding dimension is standardized with training
statistics computed federatedly (each client shares its row count, sum and sum of squares over its
training rows and all 5 views; z = (x − mean) / sqrt(var + 1e-4)); the same transform is applied to the
validation and test views. It is applied to **both** encoders so that A-frozen and B stay comparable,
and the convergence check is re-run for both. Everything else is unchanged.

**Basis of the decision.** Training loss and embedding statistics only; no validation or test
prediction of any frozen-encoder model had been computed. The first check's output is kept as
`results/linear_updates_<encoder>_raw_features.json`.
