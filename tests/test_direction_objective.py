import torch

from model.kronos import Kronos
from finetune.direction_objective import (
    DirectionAuxiliaryHead,
    compute_direction_objective,
    make_direction_targets,
)


def test_direction_targets_compare_context_end_with_requested_horizon():
    batch = torch.zeros(2, 9, 6)
    batch[0, 3, 3] = 10.0
    batch[0, 6, 3] = 12.0
    batch[1, 3, 3] = 10.0
    batch[1, 6, 3] = 8.0

    targets = make_direction_targets(
        batch,
        context_length=4,
        horizon=3,
        close_index=3,
    )

    torch.testing.assert_close(targets, torch.tensor([1.0, 0.0]))


def test_direction_objective_uses_context_boundary_hidden_state_only():
    head = DirectionAuxiliaryHead(d_model=2)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.projection.bias.zero_()

    hidden = torch.zeros(2, 8, 2)
    hidden[0, 3] = torch.tensor([4.0, 100.0])
    hidden[1, 3] = torch.tensor([-4.0, -100.0])
    hidden[:, 4:, :] = 1000.0
    batch = torch.zeros(2, 9, 6)
    batch[0, 3, 3], batch[0, 6, 3] = 10.0, 12.0
    batch[1, 3, 3], batch[1, 6, 3] = 10.0, 8.0

    result = compute_direction_objective(
        head,
        hidden,
        batch,
        context_length=4,
        horizon=3,
        close_index=3,
    )

    assert result["loss"].item() < 0.02
    assert result["accuracy"].item() == 1.0
    torch.testing.assert_close(result["logits"], torch.tensor([4.0, -4.0]))


def test_kronos_can_return_causal_context_for_auxiliary_objectives():
    model = Kronos(
        s1_bits=2,
        s2_bits=2,
        n_layers=1,
        d_model=8,
        n_heads=2,
        ff_dim=16,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        token_dropout_p=0.0,
        learn_te=False,
    )
    s1 = torch.zeros(2, 5, dtype=torch.long)
    s2 = torch.zeros(2, 5, dtype=torch.long)
    stamp = torch.zeros(2, 5, 5)

    s1_logits, s2_logits, context = model(
        s1,
        s2,
        stamp,
        return_context=True,
    )

    assert s1_logits.shape == (2, 5, 4)
    assert s2_logits.shape == (2, 5, 4)
    assert context.shape == (2, 5, 8)
