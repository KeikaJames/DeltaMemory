"""Tests for Stage 14A InfoNCE K-projector."""
from __future__ import annotations

import torch

from deltamemory.memory.k_projector import (
    InfoNCEBatch,
    KProjectorBank,
    infonce_loss,
)


def test_identity_init_returns_input_unchanged() -> None:
    proj = KProjectorBank(head_dims=[64, 128, 256])
    for layer_idx, d in enumerate([64, 128, 256]):
        x = torch.randn(7, 2, d, dtype=torch.float32)
        y = proj(layer_idx, x)
        assert y.shape == x.shape
        assert torch.allclose(y, x, atol=1e-6), f"layer {layer_idx} drifted"


def test_is_identity_is_true_at_init_and_false_after_perturb() -> None:
    proj = KProjectorBank(head_dims=[32, 32])
    assert proj.is_identity()
    with torch.no_grad():
        proj.layers[0].weight[0, 0] += 0.1
    assert not proj.is_identity()


def test_state_dict_round_trip(tmp_path) -> None:
    proj = KProjectorBank(head_dims=[16, 32])
    with torch.no_grad():
        proj.layers[0].weight.add_(0.01)
        proj.layers[1].bias.add_(0.02)
    path = tmp_path / "kproj.pt"
    proj.save(path)
    other = KProjectorBank.load(path)
    for a, b in zip(proj.layers, other.layers):
        assert torch.allclose(a.weight, b.weight)
        assert torch.allclose(a.bias, b.bias)


def test_dtype_and_device_follow_input() -> None:
    proj = KProjectorBank(head_dims=[16])
    x = torch.randn(3, 1, 16, dtype=torch.bfloat16)
    y = proj(0, x)
    assert y.dtype == torch.bfloat16
    assert y.device == x.device


def test_infonce_loss_decreases_with_aligned_pairs() -> None:
    torch.manual_seed(0)
    d = 16
    B = 32
    proj = KProjectorBank(head_dims=[d])
    # Force a non-identity start so initial loss is large and the
    # optimizer has somewhere to descend to.
    with torch.no_grad():
        proj.layers[0].weight.copy_(torch.randn(d, d) * 0.5)

    write_k = torch.randn(B, 1, d)
    query_q = write_k + 0.05 * torch.randn_like(write_k)

    batch = InfoNCEBatch(layer_idx=0, write_k=write_k, query_q=query_q)
    optim = torch.optim.AdamW(proj.parameters(), lr=1e-2)
    losses = []
    for _ in range(200):
        optim.zero_grad()
        loss = infonce_loss(proj, batch)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[0] > 1.0, f"initial loss too small: {losses[0]:.4f}"
    assert losses[-1] < losses[0] * 0.5, (
        f"InfoNCE did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"
    )


def test_attached_identity_projector_preserves_bit_equality() -> None:
    """Bank with an identity-init projector attached must still be bit-equal
    to bank with no projector at all (the central Stage 14A invariant)."""
    from deltamemory.memory.attn_native_bank import AttnNativeBank

    bank_a = AttnNativeBank(num_layers=4, num_kv_heads=2, head_dim=8)
    bank_b = AttnNativeBank(num_layers=4, num_kv_heads=2, head_dim=8)
    bank_b.k_projector = KProjectorBank(head_dims=[8, 8, 8, 8])

    K = [torch.randn(2, 8) for _ in range(4)]
    V = [torch.randn(2, 8) for _ in range(4)]
    bank_a.append(K, V, fact_id="x", address="x")
    bank_b.append([k.clone() for k in K], [v.clone() for v in V], fact_id="x", address="x")

    for layer in range(4):
        mk_a = bank_a.M_K[layer]
        mk_b = bank_b.M_K[layer]
        if bank_b.k_projector is not None:
            mk_b = bank_b.k_projector(layer, mk_b)
        assert torch.allclose(mk_a, mk_b, atol=1e-6), f"identity projector drifted at layer {layer}"
