# Eval: A-share 1/3/5/10-day multi-horizon fine-tuning

## Capability checks

- [ ] The training dataset emits raw close supervision separately from normalized model inputs.
- [ ] The auxiliary head uses only representations at or before the 128-day context boundary.
- [ ] Return and direction targets are generated for exactly 1, 3, 5, and 10 trading days.
- [ ] Direction labels are down / flat / up and use a volatility-aware, past-only dead zone.
- [ ] A checkpoint records the head architecture, four horizons, and the dead-zone contract.
- [ ] Strict backtest reports endpoint price and direction metrics separately for horizons 1, 3, 5, and 10.

## Regression checks

- [ ] Existing causal-context and data-provider tests pass.
- [ ] The 128 -> 10 manifest contract remains enforced.
- [ ] No output overwrites the prior `a_share_multi_*` checkpoints.

## Acceptance gates for the training run

- [ ] Evaluate only origins strictly after the validation boundary, with non-overlapping 10-day target blocks.
- [ ] Compare the new checkpoint with public weights, the prior multi-stock checkpoint, and last-close baselines.
- [ ] Report MAE, RMSE, MAPE, and end-point three-class direction accuracy at 1/3/5/10 days.
