"""Tests for HTWR InfoNCE projector (Exp12 T4)."""
from __future__ import annotations

import torch

from deltamemory.memory.htwr_projector import (
    InfoNCEConfig,
    LinearProjector,
    LowRankResidualProjector,
    MLPProjector,
    info_nce_loss,
    make_projector,
    train_projector,
)


def test_make_projector_kinds():
    for kind in ("linear", "shared", "lowrank", "mlp"):
        p = make_projector(kind, d_in=16, d_out=4, rank=4, d_hidden=8)
        x = torch.randn(3, 16)
        y = p(x)
        assert y.shape[0] == 3


def test_info_nce_loss_finite():
    q = torch.randn(4, 8)
    k = torch.randn(4, 8)
    pos = torch.arange(4)
    loss = info_nce_loss(q, k, pos, tau=0.07)
    assert torch.isfinite(loss)
    assert loss > 0


def test_train_projector_decreases_loss_on_separable_pairs():
    """Synthetic: queries are noisy copies of keys. Projector should learn to
    align them and pull diagonals up vs off-diagonals."""
    torch.manual_seed(0)
    n, d = 32, 24
    keys = torch.randn(n, d)
    queries = keys + 0.1 * torch.randn(n, d)
    proj = LinearProjector(d_in=d, d_out=8)
    cfg = InfoNCEConfig(tau=0.07, epochs=80, lr=5e-3, batch_size=16)
    losses = train_projector(proj, queries, keys, config=cfg)
    assert losses[-1] < losses[0]
    # Final loss should be substantially below initial (separable problem).
    assert losses[-1] < 0.5 * losses[0]


def test_lowrank_projector_is_identity_at_init():
    p = LowRankResidualProjector(d_in=16, rank=4)
    x = torch.randn(3, 16)
    y = p(x)
    # V is zero-initialized, so output == input.
    assert torch.allclose(x, y, atol=1e-6)


def test_mlp_projector_forward_shape():
    p = MLPProjector(d_in=16, d_hidden=8, d_out=4)
    x = torch.randn(2, 16)
    assert p(x).shape == (2, 4)
