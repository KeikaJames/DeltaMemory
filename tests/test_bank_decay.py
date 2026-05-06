from __future__ import annotations

import torch

from deltamemory.memory.bank_decay import apply_decay


def _state() -> dict:
    k = torch.ones(3, 1, 2)
    v = torch.ones(3, 1, 2)
    return {
        "M_K": [k],
        "M_V": [v],
        "fact_ids": ["new", "mid", "old"],
        "address_strs": ["a0", "a1", "a2"],
        "last_access_step": [100, 50, -10000],
        "write_step": [100, 50, 0],
        "original_v_norm": [2**0.5, 2**0.5, 2**0.5],
        "decay_erase_threshold": 0.01,
    }


def test_decay_shape_preserved_without_erase():
    state = _state()
    state["decay_erase_threshold"] = 0.0
    out = apply_decay(state, current_step=100, half_life=1000)
    assert out["M_V"][0].shape == state["M_V"][0].shape
    assert out["M_K"][0].shape == state["M_K"][0].shape


def test_recently_accessed_rows_not_affected():
    out = apply_decay(_state(), current_step=100, half_life=10)
    assert torch.allclose(out["M_V"][0][0], torch.ones(1, 2))


def test_very_old_rows_erased():
    out = apply_decay(_state(), current_step=100, half_life=10)
    assert out["M_V"][0].shape[0] == 2
    assert out["fact_ids"] == ["new", "mid"]


def test_alpha_zero_redline_holds_under_decay():
    logits = torch.randn(4, 8)
    out = apply_decay(_state(), current_step=100, half_life=10)
    after = logits + 0.0 * out["M_V"][0].sum()
    assert torch.equal(logits, after)
