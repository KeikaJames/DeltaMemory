"""Per-layer whitening transforms for HTWR retrieval (Exp12 T2).

All ops are pure linear algebra on CPU/fp32.  The bank residual tensor has
shape ``(N, L, D)`` so we estimate one transform per layer.

Modes:

- ``zca``    : per-layer ZCA on full D-dim residual with shrinkage λ.
- ``diag``   : per-dim variance normalization (no off-diagonals).
- ``pca``    : project to ``rank`` principal components per layer.

All transforms return ``(mu, W)`` where ``mu`` is ``(L, D)`` and ``W`` is
``(L, K, D)`` with ``K = D`` for zca/diag and ``K = rank`` for pca.  The
intended use:

    x̃ = W (x − μ)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

WhitenMode = Literal["zca", "diag", "pca"]


@dataclass
class WhitenResult:
    mu: torch.Tensor   # (L, D)
    W: torch.Tensor    # (L, K, D)
    mode: WhitenMode
    rank: int          # K
    shrinkage: float


def _layer_zca(x: torch.Tensor, shrinkage: float, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (N, D). Returns (mu, W) with W = Σ_λ^{-1/2}."""
    n, d = x.shape
    mu = x.mean(dim=0)
    xc = x - mu
    sigma = (xc.T @ xc) / max(1, n - 1)
    sigma_lam = (1.0 - shrinkage) * sigma + shrinkage * torch.eye(d, dtype=sigma.dtype)
    # eigendecomp; clamp eigenvalues for numerical safety.
    sigma_lam = 0.5 * (sigma_lam + sigma_lam.T)
    evals, evecs = torch.linalg.eigh(sigma_lam)
    evals = torch.clamp(evals, min=eps)
    inv_sqrt = evecs @ torch.diag(evals.pow(-0.5)) @ evecs.T
    return mu, inv_sqrt


def _layer_diag(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = x.shape
    mu = x.mean(dim=0)
    var = ((x - mu) ** 2).mean(dim=0).clamp(min=eps)
    W = torch.diag(var.rsqrt())
    return mu, W


def _layer_pca(x: torch.Tensor, rank: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns top-k principal directions as a (rank, D) projection matrix.

    Each row is variance-normalized so the result is whitened in the PCA subspace.
    """
    n, d = x.shape
    if rank > d:
        raise ValueError(f"pca rank {rank} > residual dim {d}")
    mu = x.mean(dim=0)
    xc = x - mu
    # SVD: xc = U S Vh.  Principal directions are rows of Vh.
    U, S, Vh = torch.linalg.svd(xc, full_matrices=False)
    # Variance of i-th PC ≈ S[i]^2 / (n-1); whiten by dividing by its std.
    stds = (S / (max(1, n - 1) ** 0.5)).clamp(min=eps)
    W = Vh[:rank] / stds[:rank].unsqueeze(-1)
    return mu, W


def fit_whitener(
    bank_keys: torch.Tensor,
    *,
    mode: WhitenMode = "zca",
    shrinkage: float = 0.1,
    rank: int | None = None,
    eps: float = 1e-6,
) -> WhitenResult:
    """Estimate a per-layer whitener from a (N, L, D) bank.

    Args:
        bank_keys: ``(N, L, D)`` residual bank, float (CPU).
        mode: ``zca``, ``diag``, or ``pca``.
        shrinkage: λ for ZCA covariance regularization, in [0, 1].
        rank: required for ``pca``.
        eps: floor for eigenvalues / variances.
    """
    if bank_keys.ndim != 3:
        raise ValueError("bank_keys must be (N, L, D).")
    n, layers, d = bank_keys.shape
    bank = bank_keys.detach().to(dtype=torch.float32, device="cpu")

    mus = []
    Ws = []
    K = (rank if mode == "pca" else d)
    if mode == "pca":
        if rank is None or rank <= 0:
            raise ValueError("pca mode requires positive rank")
    for l in range(layers):
        xl = bank[:, l, :]
        if mode == "zca":
            mu, W = _layer_zca(xl, shrinkage=shrinkage, eps=eps)
        elif mode == "diag":
            mu, W = _layer_diag(xl, eps=eps)
        elif mode == "pca":
            mu, W = _layer_pca(xl, rank=int(rank), eps=eps)
        else:
            raise ValueError(f"Unknown whitening mode: {mode}")
        mus.append(mu)
        Ws.append(W)
    return WhitenResult(
        mu=torch.stack(mus, dim=0),
        W=torch.stack(Ws, dim=0),
        mode=mode,
        rank=int(K),
        shrinkage=float(shrinkage if mode == "zca" else 0.0),
    )


__all__ = ["WhitenMode", "WhitenResult", "fit_whitener"]
