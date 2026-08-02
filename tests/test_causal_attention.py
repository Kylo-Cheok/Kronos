import torch

from model.module import MultiHeadCrossAttentionWithRoPE


def test_cross_attention_remains_causal_in_evaluation_mode():
    torch.manual_seed(7)
    attention = MultiHeadCrossAttentionWithRoPE(
        d_model=8,
        n_heads=2,
        attn_dropout_p=0.0,
        resid_dropout=0.0,
    ).eval()
    query = torch.randn(1, 3, 8)
    key = torch.randn(1, 3, 8)
    value = torch.randn(1, 3, 8)
    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:, 1:, :] += 100.0
    changed_value[:, 1:, :] -= 100.0

    original = attention(query, key, value)
    changed = attention(query, changed_key, changed_value)

    torch.testing.assert_close(original[:, 0, :], changed[:, 0, :])


def test_single_step_cross_attention_can_read_all_past_context():
    torch.manual_seed(11)
    attention = MultiHeadCrossAttentionWithRoPE(
        d_model=8,
        n_heads=2,
        attn_dropout_p=0.0,
        resid_dropout=0.0,
    ).eval()
    query = torch.randn(1, 1, 8)
    key = torch.randn(1, 3, 8)
    value = torch.randn(1, 3, 8)
    changed_value = value.clone()
    changed_value[:, 1:, :] += 100.0

    original = attention(query, key, value)
    changed = attention(query, key, changed_value)

    assert not torch.allclose(original, changed)
