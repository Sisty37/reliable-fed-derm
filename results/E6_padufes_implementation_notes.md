# E6 part 2: PAD-UFES-20 domain-shift stress test, choices fixed before predictions (6 Nov 2026)

Code: `scripts/prepare_padufes.py`, `scripts/padufes_predict.py`, `scripts/padufes_analysis.py`.
Data: PAD-UFES-20 (Pacheco et al., 2020; Mendeley Data doi:10.17632/zr7vgbcyr2.1), 2,298 smartphone
clinical images, 1,641 lesions, 1,373 patients, downloaded 6 Nov 2026 into `data/padufes20/` (not in git).

1. **Role (protocol Section 2):** a domain-shift stress test (clinical smartphone photographs instead
   of dermoscopy), not an unseen center. No model is trained or fine-tuned on PAD-UFES-20.
2. **Preprocessing:** identical to Fed-ISIC2019 (FLamby `resize_images.py`): RGB, shorter edge 224 px,
   colour constancy (power 6), JPEG; then the FLamby test transform (CenterCrop 200, Normalize).
   Model B gets the same view resized to 224 and the PanDerm CLS embedding, centred with the federated
   training mean of the same Fed-ISIC2019 split, exactly as in training.
3. **Labels:** MEL→MEL, NEV→NV, BCC→BCC, ACK→AK, SEK→BKL, SCC→SCC (protocol Section 2). All 2,298
   images belong to these six classes. Model probabilities are restricted to the six classes and
   renormalized; the Fed-ISIC2019 reference uses the same restriction on its lesion-grouped test
   images of these six classes (DF and VASC images dropped).
4. **Models:** the lesion-grouped FedAvg models A and B, seeds 0–4 (the E5 models), each with its own
   global temperature from Fed-ISIC2019 validation lesions (as in E5). Model A uses client 0's
   BatchNorm statistics (the model evaluated throughout).
5. **Metrics:** balanced accuracy (6 classes), recall per class, top-label ECE (15 bins), Brier score;
   patient bootstrap (1,000 replicates, seeds averaged inside each replicate) for PAD-UFES-20 BA and ECE.
6. **Conformal (protocol Section 5):** LAC (primary) and APS, α = 0.05, 0.1, 0.2, level
   ⌈(m+1)(1−α)⌉/m.
   (i) **Transfer:** calibration scores = all Fed-ISIC2019 lesion-grouped test images of the six
   classes (all centers pooled; the global threshold of E5 with the full test set as calibration set);
   coverage, set size and per-class coverage on all PAD-UFES-20 images; patient-bootstrap interval for
   coverage.
   (ii) **Recalibrated:** PAD-UFES-20 patients split 50/50 at random, 100 times per model; threshold
   from the calibration half; coverage (mean, 10th percentile over splits), set size and per-class
   coverage on the other half.
7. **Interpretation rule (fixed now):** transfer coverage below 1−α with the patient-bootstrap
   interval excluding 1−α is reported as a failure of calibration transfer under domain shift; the
   recalibrated result shows what a small local calibration set restores. No hypothesis test.
8. **Pretraining overlap:** PAD-UFES-20 is not listed among PanDerm's pretraining sources; as for
   ISIC, "not listed" is not proof of "not seen" (limitation; see `PanDerm_overlap_check.md`).
