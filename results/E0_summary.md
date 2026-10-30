# E0: official-split reproduction (protocol v0.10)

Run 30 Oct 2026, 05:49–13:29 (Asia/Dhaka), commit 907f3b5. Laptop with an RTX 4050 (6 GB),
PyTorch 2.5.1 + CUDA 12.1, AMP on. Model A (EfficientNet-B0), FLamby settings (commit edacf54):
9 rounds x 100 local updates, SGD lr 0.01, FedProx mu 0.001, SCAFFOLD server lr 1.0,
pooled Adam 5e-4 for 20 epochs. Seeds 0, 1, 2. Metric: balanced accuracy on the pooled official test set.

| Method | Mean (3 seeds) | Range | Mean of centers |
|---|---|---|---|
| FedAvg | 0.6148 | 0.5787–0.6415 | 0.6174 |
| FedProx | 0.6041 | 0.5666–0.6467 | 0.5966 |
| SCAFFOLD | 0.6234 | 0.6180–0.6264 | 0.6043 |
| Pooled | 0.7497 | 0.7415–0.7614 | 0.7041 |

## Gate 1

Reference (FENDA-FL, FedAvg global model): 0.657. FedAvg 0.6148 is 4.2 points below it, inside the
pre-specified ±5 margin. **Gate 1: PASS.**

## FLamby's own published values (pooled test, files in flamby/results/results_benchmark_fed_isic2019 at commit edacf54)

| Method | Seeds 42–46 | Mean of 5 | Mean without seed 46 |
|---|---|---|---|
| FedAvg | 0.581, 0.574, 0.591, 0.593, 0.330 | 0.534 | 0.585 |
| FedProx | 0.550, 0.599, 0.521, 0.606, 0.445 | 0.544 | 0.569 |
| SCAFFOLD | 0.620, 0.594, 0.549, 0.596, 0.438 | 0.559 | 0.590 |
| Pooled | 0.756, 0.776, 0.758, 0.770, 0.752 | 0.762 | 0.765 |

Our values are within about 3 points of FLamby's own results without seed 46 for all four methods,
and above them for the three federated methods. The 4.2-point gap to the Gate 1 reference therefore
comes from the reference value (FENDA-FL reports a higher FedAvg value than FLamby), not from a
pipeline difference. A line-by-line check of the configuration against FLamby edacf54
(config_isic2019.json, common.py, loss.py, model.py, the augmentations and the FedAvg aggregation in
strategies/fed_avg.py: parameter deltas weighted by client training size, BatchNorm buffers local,
stateless SGD) found no difference.

## Deviations from the protocol

1. The protocol asked for FLamby's published FedAvg value to be recorded **before** E0. It was
   recorded after E0 (30 Oct 2026). It plays no role in the Gate 1 decision, which uses the
   pre-specified FENDA-FL value of 0.657.
2. PyTorch 2.5.1 (already installed) was used instead of 2.4.1. Both are CUDA 12.1 builds.
