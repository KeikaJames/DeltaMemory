from __future__ import annotations

import torch

from deltamemory.memory.bank_importance import compute_novelty, importance_bias


def _state() -> dict:
    return {
        "M_K": [torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])],
        "M_V": [torch.ones(2, 1, 2)],
    }


def test_duplicate_has_low_novelty():
    assert compute_novelty(torch.tensor([[1.0, 0.0]]), _state()) < 1e-6


def test_unique_entry_has_high_novelty():
    assert compute_novelty(torch.tensor([[-1.0, 0.0]]), _state()) >= 1.0


def test_empty_bank_novelty_is_one():
    assert compute_novelty(torch.tensor([[1.0, 0.0]]), {"M_K": [torch.empty(0, 1, 2)]}) == 1.0


def test_importance_bias_uses_scores_and_merge_counts():
    state = _state()
    state["importance_scores"] = [0.0, 0.75]
    assert torch.allclose(importance_bias(state), torch.tensor([1.0, 1.75]))
    state.pop("importance_scores")
    state["merge_counts"] = [1.0, 4.0]
    bias = importance_bias(state)
    assert bias.shape == (2,)
    assert bias[1] > bias[0]
