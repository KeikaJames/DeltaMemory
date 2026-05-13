"""Phase X.7 — bank lifecycle (capacity / LRU evict) unit tests.

Locks H_X7.0 (default bit-equal), H_X7.1 (size cap), H_X7.2 (LRU
preserves the recently-accessed entry). PREREG: experiments/X7_forget_merge/PREREG.md.
"""
from __future__ import annotations

import torch

from deltamemory.memory.attn_native_bank import AttnNativeBank


def _mk_bank(num_layers: int = 2, num_kv_heads: int = 1, head_dim: int = 4) -> AttnNativeBank:
    return AttnNativeBank(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dims=[head_dim] * num_layers,
        device="cpu",
        dtype=torch.float32,
    )


def _per_layer(bank: AttnNativeBank, val: float) -> tuple[list, list]:
    K = [torch.full((bank.num_kv_heads, bank.head_dims[layer_idx]), val) for layer_idx in range(bank.num_layers)]
    V = [torch.full((bank.num_kv_heads, bank.head_dims[layer_idx]), val) for layer_idx in range(bank.num_layers)]
    return K, V


def test_x7_capacity_default_bit_equal_append():
    """H_X7.0 — bank_capacity=0 (default) ⇒ append never evicts."""
    bank = _mk_bank()
    assert bank.bank_capacity == 0
    for i in range(50):
        K, V = _per_layer(bank, float(i))
        bank.append(K, V, fact_id=f"f{i}", address=f"a{i}")
    assert bank.size == 50
    assert bank.fact_ids == [f"f{i}" for i in range(50)]


def test_x7_capacity_default_bit_equal_bulk():
    """H_X7.0 — bulk_append with no capacity ⇒ no eviction."""
    bank = _mk_bank()
    n = 30
    K = [torch.arange(n * bank.head_dims[layer_idx], dtype=torch.float32).reshape(n, bank.num_kv_heads, bank.head_dims[layer_idx]) for layer_idx in range(bank.num_layers)]
    V = [k.clone() for k in K]
    bank.bulk_append(K, V, [f"f{i}" for i in range(n)], [f"a{i}" for i in range(n)])
    assert bank.size == n


def test_x7_lru_evicts_to_capacity_append():
    """H_X7.1 — bank_capacity=K caps size at K after N>K appends."""
    bank = _mk_bank()
    bank.bank_capacity = 5
    bank.bank_evict_policy = "lru"
    for i in range(20):
        K, V = _per_layer(bank, float(i))
        bank.append(K, V, fact_id=f"f{i}", address=f"a{i}")
    assert bank.size == 5
    # Pure-LRU + no reads ⇒ keep the 5 most-recently-written.
    assert bank.fact_ids == [f"f{i}" for i in range(15, 20)]


def test_x7_fifo_evicts_oldest():
    """FIFO policy ⇒ keeps most recently inserted by index."""
    bank = _mk_bank()
    bank.bank_capacity = 3
    bank.bank_evict_policy = "fifo"
    for i in range(10):
        K, V = _per_layer(bank, float(i))
        bank.append(K, V, fact_id=f"f{i}", address=f"a{i}")
    assert bank.size == 3
    assert bank.fact_ids == ["f7", "f8", "f9"]


def test_x7_lru_preserves_accessed_entry():
    """H_X7.2 — LRU keeps an entry that was *read* recently even if it was written first."""
    bank = _mk_bank()
    bank.bank_capacity = 3
    bank.bank_evict_policy = "lru"
    # Write 4 entries; bank caps at 3 so first one evicts.
    for i in range(4):
        K, V = _per_layer(bank, float(i))
        bank.append(K, V, fact_id=f"f{i}", address=f"a{i}")
    assert bank.fact_ids == ["f1", "f2", "f3"]
    # Now re-touch f1 (currently the oldest) → it should survive the next eviction.
    idx_f1 = bank.fact_ids.index("f1")
    bank._x7_note_access([idx_f1])
    # Append two more → must evict 2.
    for i in range(4, 6):
        K, V = _per_layer(bank, float(i))
        bank.append(K, V, fact_id=f"f{i}", address=f"a{i}")
    assert bank.size == 3
    assert "f1" in bank.fact_ids  # protected by recent access
    assert "f2" not in bank.fact_ids and "f3" not in bank.fact_ids


def test_x7_eviction_keeps_layers_aligned():
    """Per-layer M_K / M_V must stay row-aligned with fact_ids after eviction."""
    bank = _mk_bank(num_layers=3, head_dim=4)
    bank.bank_capacity = 2
    for i in range(5):
        K, V = _per_layer(bank, float(i + 1))
        bank.append(K, V, fact_id=f"f{i}", address=f"a{i}")
    assert bank.size == 2
    for layer in range(bank.num_layers):
        assert bank.M_K[layer].size(0) == 2
        assert bank.M_V[layer].size(0) == 2
        # Surviving entries are f3 (val=4) and f4 (val=5).
        assert torch.allclose(bank.M_K[layer][0], torch.full_like(bank.M_K[layer][0], 4.0))
        assert torch.allclose(bank.M_K[layer][1], torch.full_like(bank.M_K[layer][1], 5.0))
