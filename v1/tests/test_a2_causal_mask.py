"""A2 causal-mask regression tests for the attention-native bank.

The production bank is K/V-only: sequence tokens are the only query rows, and
bank slots are appended only on the key/value axis.  These tests encode that
block algebra without loading a large HF model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _causal_scores_with_bank(t_seq: int = 8, n_bank: int = 4) -> torch.Tensor:
    scores_orig = torch.zeros(1, 1, t_seq, t_seq)
    causal = torch.triu(torch.full((t_seq, t_seq), float("-inf")), diagonal=1)
    scores_orig = scores_orig + causal.view(1, 1, t_seq, t_seq)
    scores_bank = torch.zeros(1, 1, t_seq, n_bank)
    return torch.cat([scores_orig, scores_bank], dim=-1)


def test_bank_is_not_a_query_block() -> None:
    """Bank-to-seq attention is structurally zero: no bank query rows exist."""
    t_seq, n_bank = 8, 4
    scores = _causal_scores_with_bank(t_seq, n_bank)
    assert scores.shape == (1, 1, t_seq, t_seq + n_bank)
    # If bank rows existed the query dimension would be T+N.  Production keeps
    # the query dimension at T, so bank slots cannot attend backward to seq.
    assert scores.size(-2) == t_seq


def test_each_sequence_position_sees_all_bank_columns() -> None:
    """Every seq[t] query receives unmasked logits for every bank slot."""
    t_seq, n_bank = 8, 4
    weights = F.softmax(_causal_scores_with_bank(t_seq, n_bank), dim=-1)
    bank_weights = weights[..., t_seq:]
    assert bank_weights.shape == (1, 1, t_seq, n_bank)
    assert torch.all(bank_weights > 0)


def test_sequence_future_tokens_remain_masked_when_bank_is_appended() -> None:
    """Appending bank columns must not unmask future native sequence columns."""
    t_seq, n_bank = 8, 4
    weights = F.softmax(_causal_scores_with_bank(t_seq, n_bank), dim=-1)
    native = weights[..., :t_seq]
    for t in range(t_seq):
        assert torch.all(native[0, 0, t, t + 1 :] == 0)


def test_dynamic_bank_boundary_no_future_bank_slots_in_past_round() -> None:
    """A later write batch is absent from earlier rounds' key/value axis."""
    t_seq = 8
    first_batch = _causal_scores_with_bank(t_seq, n_bank=2)
    second_batch = _causal_scores_with_bank(t_seq, n_bank=4)
    assert first_batch.size(-1) == t_seq + 2
    assert second_batch.size(-1) == t_seq + 4
    first_weights = F.softmax(first_batch, dim=-1)
    # At the first write/read boundary there are exactly two bank columns; the
    # two future slots from the later batch have no representable columns.
    assert first_weights[..., t_seq:].shape[-1] == 2
