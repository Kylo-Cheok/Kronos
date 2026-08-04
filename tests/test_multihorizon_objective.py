import torch

from finetune.multihorizon_objective import (
    MultiHorizonForecastHead,
    compute_multihorizon_objective,
    make_multihorizon_targets,
)


def test_multihorizon_targets_use_raw_close_and_requested_endpoints():
    raw_close = torch.tensor(
        [[100.0, 101.0, 102.0, 102.1, 102.6, 101.4, 103.0]],
        dtype=torch.float32,
    )

    targets = make_multihorizon_targets(
        raw_close,
        context_length=3,
        horizons=(1, 3, 4),
        min_deadzone=0.005,
        volatility_multiplier=0.0,
    )

    expected_returns = torch.log(torch.tensor([[102.1, 101.4, 103.0]]) / 102.0)
    torch.testing.assert_close(targets["returns"], expected_returns)
    assert targets["direction"].tolist() == [[1, 0, 2]]


def test_multihorizon_head_never_pools_future_hidden_states():
    torch.manual_seed(7)
    head = MultiHorizonForecastHead(d_model=4, horizons=(1, 3), pool_size=3)
    head.eval()
    hidden = torch.randn(2, 8, 4)

    baseline = head(hidden, context_length=4)
    hidden[:, 4:, :] = 10000.0
    changed_future = head(hidden, context_length=4)

    torch.testing.assert_close(baseline["return_prediction"], changed_future["return_prediction"])
    torch.testing.assert_close(baseline["direction_logits"], changed_future["direction_logits"])


def test_multihorizon_objective_has_one_loss_and_accuracy_per_horizon():
    torch.manual_seed(11)
    head = MultiHorizonForecastHead(d_model=3, horizons=(1, 3), pool_size=2)
    hidden = torch.randn(2, 7, 3)
    raw_close = torch.tensor(
        [
            [10.0, 10.1, 10.2, 10.4, 10.5, 10.6, 10.7],
            [20.0, 19.9, 19.8, 19.7, 19.5, 19.3, 19.2],
        ]
    )

    result = compute_multihorizon_objective(
        head,
        hidden,
        raw_close,
        context_length=4,
        horizons=(1, 3),
        min_deadzone=0.001,
        volatility_multiplier=0.0,
    )

    assert result["return_loss"].ndim == 0
    assert result["direction_loss"].ndim == 0
    assert result["direction_accuracy_by_horizon"].shape == (2,)
    assert result["targets"]["direction"].shape == (2, 2)
