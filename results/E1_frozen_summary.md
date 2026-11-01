# E1, models A-frozen and B: inflation and analysis 2f (protocol v0.10)

Linear heads trained 1–2 Nov 2026, commit 16c2f2e (settings fixed before any test result: tags
`e1-frozen-final` and commit 16c2f2e; centred embeddings, U = 6400 local updates per round for both
encoders). Analysis `scripts/gate2.py --prefix`, 2,000 lesion-bootstrap replicates; full output in
`results/E1_gate2_effb0frozen.json` and `results/E1_gate2_pandermfrozen.json`. Same splits, test sets
and new-lesion subsets as model A. Model B: pretraining overlap not ruled out.

## Inflation (BA on the full image-level test set minus reweighted BA on its new-lesion subset)

| Model | FedAvg, 9 rounds | Pooled, 20 epochs |
|---|---|---|
| A, full fine-tuning | +8.11 [+4.74, +11.18], seeds +2.43 to +13.31 | +16.16 [+12.50, +19.42], seeds +12.18 to +19.90 |
| A-frozen, EfficientNet-B0 + linear head | +8.21 [+4.96, +11.47], seeds +5.87 to +10.71 | +6.11 [+2.46, +9.78], seeds +4.29 to +7.27 |
| B, PanDerm-Base + linear head | +8.69 [+5.23, +12.63], seeds +6.38 to +13.26 | +7.35 [+3.64, +11.53], seeds +3.82 to +13.36 |

(Model A FedAvg at 27 rounds: +12.41 [+9.61, +15.62].)

## Between-condition difference (sensitivity; BA of the image-level model on its own test set minus BA of the lesion-grouped model on its own test set)

| Model | FedAvg | Pooled |
|---|---|---|
| A | 0.5773 − 0.5086 = +6.87 | 0.7274 − 0.5756 = +15.18 |
| A-frozen | 0.5048 − 0.4298 = +7.49 | 0.5747 − 0.5150 = +5.98 |
| B | 0.5871 − 0.5248 = +6.23 | 0.5961 − 0.5576 = +3.85 |

## Melanoma recall, full test set vs new-lesion subset (FedAvg, seeds 0/1/2)

- A-frozen: 0.463/0.381, 0.464/0.378, 0.474/0.331
- B: 0.524/0.386, 0.537/0.373, 0.550/0.389

## Reading (secondary, no decision attached; 3 seeds)

1. **Leakage inflates every model.** All six inflation intervals lie above zero, and the
   between-condition check points the same way in every case.
2. **Analysis 2f, primary contrast (A vs A-frozen).** Under FedAvg the inflation does not depend on
   the adaptation regime (+8.1 vs +8.2). Under pooled training, full fine-tuning inflates far more
   (+16.2 vs +6.1). Only the strongly fitted full network shows the extra, regime-dependent part.
3. **Interpretation.** A frozen encoder with a linear head can memorize little, yet it still gains
   about 6–9 points from leakage. The likely route is proximity: images of the same lesion lie close
   together in feature space, so a test image whose sibling is in training is classified like that
   sibling even without memorization. On top of this, a fully fine-tuned network that fits its
   training images closely (pooled, or FedAvg at 27 rounds) gains more. This reading is
   descriptive; a direct check (how often a test image's nearest training neighbour is its own
   sibling) would be exploratory and is not part of the protocol.
4. **Model B** behaves like A-frozen (FedAvg +8.7, pooled +7.4), so a stronger,
   dermatology-specific frozen encoder does not remove the problem. B's overlap status limits how
   far this can be generalized.
5. **Threat (Section 3).** If new-lesion images were simply harder, inflation would appear even
   without leakage, and frozen models would show it too. The between-condition difference does not
   use the new-lesion subset and still gives +3.9 to +7.5 points for the frozen models, which argues
   against a pure difficulty effect, but the two test sets differ in make-up, so this is partial
   evidence only.
