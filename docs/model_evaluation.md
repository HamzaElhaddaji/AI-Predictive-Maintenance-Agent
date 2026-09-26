## ML Evaluation

## Objective
Evaluate the trained RUL (Remaining Useful Life) prediction model beyond simple accuracy, and define an operational decision threshold for maintenance alerts.

## Models Compared

| Model | RMSE | MAE | R2 |
|---|---|---|---|
| Random Forest | 17.77 | 13.62 | 0.820 |
| **XGBoost (selected)** | **16.86** | **12.65** | **0.838** |

XGBoost was selected as the final model based on lower RMSE/MAE and higher R2.

## Error Analysis

- RUL predictions are clipped to a maximum of 125 cycles (piecewise linear RUL), matching the standard approach for this dataset family.
- Error was analyzed across RUL ranges (0-20, 20-50, 50-80, 80-125 cycles) to check that the model remains accurate near failure, which is the operationally critical zone.
- Optimistic predictions (predicted RUL > true RUL) were checked specifically, since overestimating remaining life is more operationally dangerous than underestimating it.

## PHM08 Competition Score

An asymmetric scoring function (as used in the original PHM08 challenge) was computed to penalize late/optimistic predictions more heavily than early ones:

- **PHM08 score: 52195.44** (lower is better)

This score is not directly comparable to a simple RMSE — it reflects the operational cost of being wrong in each direction.

## Operational Decision Threshold

The regression output was converted into a binary maintenance-alert decision: **alert if predicted RUL < 30 cycles.**

| Metric | Value |
|---|---|
| Precision@30 | 0.880 |
| Recall@30 | 0.840 |
| F1@30 | 0.860 |

**Interpretation:**
- Precision (0.88): when the model raises an alert, it is correct 88% of the time (low false-alarm rate).
- Recall (0.84): the model catches 84% of machines that are actually within 30 cycles of failure.
- This threshold offers a reasonable trade-off between avoiding unnecessary maintenance and catching real failures early.

## Deliverable Status

- [x] Trained model documented with RMSE/MAE/R2
- [x] Model compared against a baseline (Random Forest)
- [x] Asymmetric error behavior analyzed
- [x] PHM08-style score computed
- [x] Operational alert threshold defined with precision/recall/F1
- [x] Final model saved to `models/xgb_rul_model.pkl`

## Next Steps 

Expose the trained model as a callable tool (`predict_machine_failure()`) so it can be used by the AI agent in later phases.
