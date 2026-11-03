# E5 / E5b: implementation choices fixed before running `scripts/e5_conformal.py` (3 Nov 2026)

Protocol v0.10, Section 5 defines the conformal methods, ShrinkFCP, the RQ4 metric, the reference and
the success criterion. Choices it leaves open:

1. **Models.** The lesion-grouped FedAvg models (9 rounds) of model A and model B, seeds 0–4 (from E1
   and E2). FedAvg is the global federated model whose thresholds the aggregation strategies share.
   Leave-one-center-out models (E6) follow later.
2. **Primary score: LAC**, α = 0.1. APS and α = 0.05 / 0.2 are reported without a bootstrap test.
3. **Temperature.** Each client fits a temperature on its own validation lesions (NLL, grid of 600
   values between 0.05 and 20); the global temperature is their mean weighted by validation images
   and is applied to all scores (calibration and test).
4. **APS** is the non-randomized version (mass of all classes ranked at or above the true class);
   prediction sets are all classes with score ≤ threshold, empty sets allowed, for both scores.
5. **Federated quantiles** (global, FCP, and q_global inside ShrinkFCP) use 1,000-bin histograms on
   [0, 1] that each client shares; the threshold is the upper edge of the bin that reaches the
   required rank (conservative by at most 0.001). Client-wise thresholds use the exact local quantile.
6. **FCP** uses the rank ⌈(m + K)(1 − α)⌉ of m pooled calibration scores from K = 6 clients, as in the
   authors' code (github.com/clu5/federated-conformal, `get_distributed_quantile`).
7. **Personalized FCP (Min et al., 2025) is not included.** Its threshold rule could not be verified
   from the NeurIPS 2025 abstract and no code was found. It is a comparison only (not part of the RQ4
   reference or success criterion); this is a deviation from the protocol, to be revisited if the
   paper's method can be read in full.
8. **ShrinkFCP primary k:** median over centers of n_c in each calibration draw (n_c = calibration
   lesions of center c). Sensitivity: k grid {0, 10, 25, 50, 100, 200, 500, ∞}; data-driven k chosen
   per seed on validation lesions (50/50 split by lesion, 100 repeats, objective = worst-center
   lower-tail coverage; ties go to the smaller k), then applied to the test set.
9. **Seeds.** Every statistic is computed per seed over 100 calibration/evaluation splits and averaged
   over the 5 seeds. Bootstrap: 2,000 replicates, each resampling every seed's test lesions with
   replacement (a resampled lesion counts as a new lesion when the halves are drawn) and redrawing 20
   splits shared by ShrinkFCP and the reference; the reference (global or client-wise) is fixed from
   the point estimate.
10. The script was checked on synthetic predictions (centers with different calibration) before being
    run on the real models.
