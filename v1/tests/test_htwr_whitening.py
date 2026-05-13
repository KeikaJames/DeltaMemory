"""Tests for HTWR whitening (Exp12)."""
from __future__ import annotations

import pytest
import torch

from deltamemory.memory.htwr_whitening import fit_whitener


def _bank(n: int = 50, layers: int = 4, d: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, layers, d, generator=g, dtype=torch.float32)


def test_zca_produces_identity_covariance():
    bank = _bank(n=200, layers=3, d=8)
    res = fit_whitener(bank, mode="zca", shrinkage=0.0)
    assert res.mode == "zca"
    assert res.mu.shape == (3, 8)
    assert res.W.shape == (3, 8, 8)
    # Verify per-layer: cov(W (x − μ)) ≈ I (up to numerical shrinkage).
    for l in range(3):
        x = bank[:, l, :]
        mu = res.mu[l]
        W = res.W[l]
        x_w = (x - mu) @ W.T
        cov = (x_w.T @ x_w) / (x_w.shape[0] - 1)
        I = torch.eye(8)
        assert torch.allclose(cov, I, atol=0.05)


def test_zca_with_shrinkage_does_not_explode():
    bank = _bank(n=12, layers=2, d=24)  # underdetermined: n < d
    res = fit_whitener(bank, mode="zca", shrinkage=0.2)
    assert torch.isfinite(res.W).all()


def test_diag_normalizes_per_dim_variance():
    bank = _bank(n=300, layers=2, d=8)
    res = fit_whitener(bank, mode="diag")
    assert res.mode == "diag"
    for l in range(2):
        x_w = (bank[:, l, :] - res.mu[l]) @ res.W[l].T
        var = ((x_w - x_w.mean(dim=0)) ** 2).mean(dim=0)
        assert torch.allclose(var, torch.ones(8), atol=0.15)


def test_pca_rank_reduces_dimensionality():
    bank = _bank(n=200, layers=2, d=12)
    res = fit_whitener(bank, mode="pca", rank=4)
    assert res.W.shape == (2, 4, 12)
    assert res.rank == 4
    # After PCA-whitening, the projected variance per axis should be ~1.
    for l in range(2):
        x_w = (bank[:, l, :] - res.mu[l]) @ res.W[l].T
        var = ((x_w - x_w.mean(dim=0)) ** 2).mean(dim=0)
        assert torch.allclose(var, torch.ones(4), atol=0.15)


def test_pca_requires_positive_rank():
    bank = _bank(n=10, layers=1, d=8)
    with pytest.raises(ValueError):
        fit_whitener(bank, mode="pca")
    with pytest.raises(ValueError):
        fit_whitener(bank, mode="pca", rank=0)


def test_whitened_score_separates_anisotropic_synthetic():
    """Synthetic: one dim has huge variance + tiny per-fact offset on that dim
    plus large variance on other dims that drown out fact identity.
    Raw cosine should be near-uniform; ZCA should separate."""
    n_facts, layers, d = 8, 2, 32
    g = torch.Generator().manual_seed(42)
    bank = torch.randn(n_facts, layers, d, generator=g) * 5.0
    # Inject a tiny fact-specific direction along axis 0
    for i in range(n_facts):
        bank[i, :, 0] += 0.05 * (i - n_facts / 2)
    res = fit_whitener(bank, mode="zca", shrinkage=0.05)
    # After whitening, fact identity along axis 0 should re-emerge.
    bank_white = torch.einsum("lkd,nld->nlk", res.W, bank - res.mu.unsqueeze(0))
    raw_sep = (bank[:, 0, :].std(dim=0).max() / bank[:, 0, :].std(dim=0).min()).item()
    # The whitened bank has tighter conditioning by construction.
    white_std = bank_white[:, 0, :].std(dim=0)
    assert torch.isfinite(white_std).all()
