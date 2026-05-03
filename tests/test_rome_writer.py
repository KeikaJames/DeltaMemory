"""Tests for Stage 14E ROME-style writer rebuild and Stage 14D temperature."""
from __future__ import annotations

import pytest
import torch

from deltamemory.memory.attn_native_bank import AttnNativeBank
from deltamemory.memory.rome_writer import (
    rebuild_bank_v,
    solve_rome_writer,
)


def test_solve_rome_writer_perfectly_recovers_linear_map() -> None:
    torch.manual_seed(0)
    d = 16
    n = 32
    W_true = torch.randn(d, d) * 0.5
    K = torch.randn(n, d)
    V = K @ W_true
    W, stats = solve_rome_writer(K, V, lambda_reg=1e-6)
    pred = K @ W
    err = (pred - V).pow(2).mean().sqrt().item()
    assert err < 1e-3, f"residual too large: {err}"
    assert stats.n_facts == n


def test_solve_rome_writer_handles_underdetermined_via_ridge() -> None:
    torch.manual_seed(0)
    d = 16
    n = 4  # n < d -> KᵀK is rank-deficient; ridge must rescue.
    K = torch.randn(n, d)
    V = torch.randn(n, d)
    W, stats = solve_rome_writer(K, V, lambda_reg=1e-2)
    assert W.shape == (d, d)
    assert stats.residual_rms >= 0


def test_solve_rome_writer_rejects_bad_inputs() -> None:
    K = torch.randn(4, 8)
    with pytest.raises(ValueError):
        solve_rome_writer(K, torch.randn(4, 16), lambda_reg=1e-2)
    with pytest.raises(ValueError):
        solve_rome_writer(K, torch.randn(4, 8), lambda_reg=0.0)
    with pytest.raises(ValueError):
        solve_rome_writer(torch.randn(2, 3, 4), torch.randn(2, 3, 4), lambda_reg=1e-2)


def test_rebuild_bank_v_lowers_residual_on_self_consistent_pairs() -> None:
    torch.manual_seed(0)
    bank = AttnNativeBank(num_layers=3, num_kv_heads=2, head_dim=8)
    n_facts = 16
    for i in range(n_facts):
        K = [torch.randn(2, 8) for _ in range(3)]
        V = [k.clone() * 0.7 + 0.1 for k in K]   # V is a linear function of K
        bank.append(K, V, fact_id=f"f{i}", address=f"a{i}")
    pre_v = [v.clone() for v in bank.M_V]
    stats = rebuild_bank_v(bank, lambda_reg=1e-3)
    for layer in range(3):
        for h in range(2):
            assert stats[layer][h].residual_rms < 0.1
    for old, new in zip(pre_v, bank.M_V):
        assert old.shape == new.shape
        assert not torch.allclose(old, new), "bank V should change after rebuild"


def test_rebuild_bank_v_no_facts_is_noop() -> None:
    bank = AttnNativeBank(num_layers=2, num_kv_heads=1, head_dim=4)
    stats = rebuild_bank_v(bank, lambda_reg=1e-2)
    assert stats == [[], []]


def test_bank_temperature_default_is_one_and_attribute_settable() -> None:
    bank = AttnNativeBank(num_layers=1, num_kv_heads=1, head_dim=4)
    assert bank.bank_temperature == 1.0
    bank.bank_temperature = 0.5
    assert bank.bank_temperature == 0.5
