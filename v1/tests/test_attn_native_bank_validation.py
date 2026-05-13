"""Validation redlines for AttnNativeBank.append / bulk_append.

Audit findings H1+H2 (2026-05-05): public write APIs previously checked
only K layer count; mismatched V layer count or fact_ids/addresses length
mismatch could silently corrupt bank metadata.
"""
from __future__ import annotations

import pytest
import torch

from deltamemory.memory.attn_native_bank import AttnNativeBank


def _bank(num_layers=2, num_kv_heads=2, head_dim=4, head_dims=None):
    return AttnNativeBank(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dims=head_dims,
        device="cpu",
        dtype=torch.float32,
    )


def _layer_kv(num_layers=2, n=1, num_kv_heads=2, head_dim=4):
    K = [torch.randn(n, num_kv_heads, head_dim) for _ in range(num_layers)]
    V = [torch.randn(n, num_kv_heads, head_dim) for _ in range(num_layers)]
    return K, V


def test_append_rejects_short_V_layers():
    bank = _bank()
    K, V = _layer_kv()
    V_short = V[:-1]  # one layer missing
    # append takes [num_kv_heads, head_dim] not [n, ...]; reshape accordingly
    K1 = [k.squeeze(0) for k in K]
    V1 = [v.squeeze(0) for v in V_short]
    with pytest.raises(ValueError, match="layer V"):
        bank.append(K1, V1, fact_id="x", address="x")


def test_bulk_append_rejects_short_V_layers():
    bank = _bank()
    K, V = _layer_kv(n=2)
    V_short = V[:-1]
    with pytest.raises(ValueError, match="layer V"):
        bank.bulk_append(K, V_short, fact_ids=["a", "b"], addresses=["a", "b"])


def test_bulk_append_rejects_address_length_mismatch():
    bank = _bank()
    K, V = _layer_kv(n=2)
    with pytest.raises(ValueError, match="addresses"):
        bank.bulk_append(K, V, fact_ids=["a", "b"], addresses=["only_one"])
    # Bank metadata must remain consistent after the rejection.
    assert bank.size == 0
    assert len(bank.fact_ids) == len(bank.address_strs) == 0


def test_bulk_append_rejects_batch_dim_mismatch():
    bank = _bank()
    K, V = _layer_kv(n=2)
    # Inject a 3-row tensor on layer 1; fact_ids says n=2.
    K_bad = [K[0], torch.randn(3, 2, 4)]
    with pytest.raises(ValueError, match="shape mismatch"):
        bank.bulk_append(K_bad, V, fact_ids=["a", "b"], addresses=["a", "b"])


def test_append_rejects_same_numel_wrong_shape():
    bank = _bank(num_layers=1, num_kv_heads=2, head_dim=4)
    bad = torch.randn(4, 2)
    good = torch.randn(2, 4)
    with pytest.raises(ValueError, match="K shape mismatch"):
        bank.append([bad], [good], fact_id="x", address="x")
    assert bank.size == 0


def test_bulk_append_rejects_same_numel_wrong_shape():
    bank = _bank(num_layers=1, num_kv_heads=2, head_dim=4)
    bad = torch.randn(2, 4, 2)
    good = torch.randn(2, 2, 4)
    with pytest.raises(ValueError, match="K shape mismatch"):
        bank.bulk_append([bad], [good], fact_ids=["a", "b"], addresses=["a", "b"])
    assert bank.size == 0


def test_clear_resets_lopi_state():
    bank = _bank(num_layers=2, num_kv_heads=2, head_dim=4)
    bank.lopi_state.prev_q_per_layer[0] = torch.ones(1)
    bank.lopi_state.prev_residual_norms = {0: 1.0, 1: 2.0}
    bank.lopi_state.pending_residual_norms = {0: 3.0, 1: 4.0}
    bank.clear()
    assert bank.lopi_state.prev_q_per_layer == {}
    assert bank.lopi_state.prev_residual_norms == {}
    assert bank.lopi_state.pending_residual_norms == {}


def test_bulk_append_succeeds_when_consistent():
    bank = _bank()
    K, V = _layer_kv(n=2)
    bank.bulk_append(K, V, fact_ids=["a", "b"], addresses=["addr_a", "addr_b"])
    assert bank.size == 2
    assert bank.fact_ids == ["a", "b"]
    assert bank.address_strs == ["addr_a", "addr_b"]
