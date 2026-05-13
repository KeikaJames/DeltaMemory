"""Unit tests for Exp14+15 subbank operations.

Uses a synthetic AttnNativeBank — no model load required.  Validates:
  * subbank_select preserves shapes and fact_ids
  * subbank_correct returns a 1-slot bank with the correct fact
  * subbank_random excludes given fact_ids
  * subbank_swap_KV mixes K from one bank with V from another
  * subbank_shuffle_layer / _head / _V preserve per-tensor norms
  * subbank_mask_layers / _heads zero only the excluded regions
"""
from __future__ import annotations

import torch

from deltamemory.memory.attn_native_bank import AttnNativeBank
from deltamemory.memory.anb_addressed import (
    subbank_select, subbank_correct, subbank_random,
    subbank_swap_KV, subbank_shuffle_layer, subbank_shuffle_head,
    subbank_shuffle_V, subbank_mask_layers, subbank_mask_heads,
)


def _make_bank(num_layers=3, num_kv_heads=2, head_dim=4, n_facts=5,
               device="cpu", dtype=torch.float32) -> AttnNativeBank:
    bank = AttnNativeBank(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        head_dims=[head_dim] * num_layers,
        num_kv_heads_per_layer=[num_kv_heads] * num_layers,
        device=device,
        dtype=dtype,
    )
    g = torch.Generator().manual_seed(0)
    for i in range(n_facts):
        K = [torch.randn(num_kv_heads, head_dim, generator=g) for _ in range(num_layers)]
        V = [torch.randn(num_kv_heads, head_dim, generator=g) for _ in range(num_layers)]
        bank.append(K, V, fact_id=f"f{i}", address=f"addr{i}")
    return bank


def test_subbank_select_preserves_shapes():
    bank = _make_bank()
    sub = subbank_select(bank, [1, 3])
    assert sub.fact_ids == ["f1", "f3"]
    for layer in range(bank.num_layers):
        assert sub.M_K[layer].shape == (2, 2, 4)
        assert sub.M_V[layer].shape == (2, 2, 4)
        assert torch.equal(sub.M_K[layer][0], bank.M_K[layer][1])
        assert torch.equal(sub.M_K[layer][1], bank.M_K[layer][3])


def test_subbank_correct_one_slot():
    bank = _make_bank()
    sub = subbank_correct(bank, "f2")
    assert sub.fact_ids == ["f2"]
    assert sub.M_K[0].shape == (1, 2, 4)
    assert torch.equal(sub.M_K[1][0], bank.M_K[1][2])


def test_subbank_random_excludes():
    bank = _make_bank()
    rng = torch.Generator().manual_seed(7)
    sub = subbank_random(bank, rng, k=3, exclude=["f0", "f4"])
    assert len(sub.fact_ids) == 3
    assert "f0" not in sub.fact_ids
    assert "f4" not in sub.fact_ids


def test_subbank_swap_KV_takes_K_from_a_V_from_b():
    bank_a = _make_bank()
    bank_b = _make_bank()
    # Construct a bank where V is replaced from b.
    sub_a = subbank_select(bank_a, [0, 1])
    sub_b = subbank_select(bank_b, [3, 4])
    mix = subbank_swap_KV(sub_a, sub_b)
    assert mix.fact_ids == ["f0", "f1"]  # identity from K source
    for layer in range(bank_a.num_layers):
        assert torch.equal(mix.M_K[layer], sub_a.M_K[layer])
        assert torch.equal(mix.M_V[layer], sub_b.M_V[layer])


def test_subbank_shuffle_layer_preserves_per_layer_norm():
    bank = _make_bank()
    rng = torch.Generator().manual_seed(11)
    sh = subbank_shuffle_layer(bank, rng)
    # Total Frobenius norm summed across layers is preserved by a layer permutation.
    n_before = sum(b.norm().item() for b in bank.M_K)
    n_after = sum(b.norm().item() for b in sh.M_K)
    assert abs(n_before - n_after) < 1e-4
    # And the multiset of per-layer norms matches.
    nb = sorted(b.norm().item() for b in bank.M_K)
    na = sorted(b.norm().item() for b in sh.M_K)
    for x, y in zip(nb, na):
        assert abs(x - y) < 1e-4


def test_subbank_shuffle_head_preserves_per_layer_norm():
    bank = _make_bank()
    rng = torch.Generator().manual_seed(13)
    sh = subbank_shuffle_head(bank, rng)
    for layer in range(bank.num_layers):
        assert abs(bank.M_K[layer].norm().item() - sh.M_K[layer].norm().item()) < 1e-5
        assert abs(bank.M_V[layer].norm().item() - sh.M_V[layer].norm().item()) < 1e-5


def test_subbank_shuffle_V_keeps_K_unchanged():
    bank = _make_bank()
    rng = torch.Generator().manual_seed(17)
    sh = subbank_shuffle_V(bank, rng)
    for layer in range(bank.num_layers):
        assert torch.equal(bank.M_K[layer], sh.M_K[layer])
    # V is permuted (same per-slot multiset across layers).
    sb = sorted(v.norm().item() for v in bank.M_V[0])
    sa = sorted(v.norm().item() for v in sh.M_V[0])
    for x, y in zip(sb, sa):
        assert abs(x - y) < 1e-5


def test_mask_layers_zeros_excluded_only():
    bank = _make_bank(num_layers=4)
    masked = subbank_mask_layers(bank, keep_layers=[1, 2])
    for layer in [0, 3]:
        assert masked.M_K[layer].abs().sum().item() == 0.0
        assert masked.M_V[layer].abs().sum().item() == 0.0
    for layer in [1, 2]:
        assert torch.equal(masked.M_K[layer], bank.M_K[layer])
        assert torch.equal(masked.M_V[layer], bank.M_V[layer])


def test_mask_heads_zeros_only_excluded_heads():
    bank = _make_bank(num_kv_heads=4)
    masked = subbank_mask_heads(bank, layer_head_keep={0: [1, 3], 1: []})
    # Layer 0: heads 1 and 3 kept, 0 and 2 zeroed.
    assert masked.M_K[0][:, 0, :].abs().sum().item() == 0.0
    assert masked.M_K[0][:, 2, :].abs().sum().item() == 0.0
    assert torch.equal(masked.M_K[0][:, 1, :], bank.M_K[0][:, 1, :])
    # Layer 1: all heads zeroed.
    assert masked.M_K[1].abs().sum().item() == 0.0
    assert masked.M_V[1].abs().sum().item() == 0.0
    # Layer 2: untouched (not in dict).
    assert torch.equal(masked.M_K[2], bank.M_K[2])
