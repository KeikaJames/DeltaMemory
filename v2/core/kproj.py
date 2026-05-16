"""Low-rank residual K/V projector for AttentionBank reads.

Phase B2 result (v1/experiments/atb_validation_v1/exp42_lpl/06_phase_b2_kproj.py):
adding a learnable (I + P) projection on bank h-vectors before they go through
the frozen W_K / W_V drops Qwen3-4B test NLL from 12.13 to 6.30 (random-bank
control: -0.02). Without P the static-bank bridge is null. This module
extracts the projector for v2 reuse.

Usage::

    P = LowRankProj(d=hidden_size, r=64).to(device).float()
    # later, when assembling bank entries before they enter attention:
    h_bank_proj = h_bank + P(h_bank)        # residual: P init zero -> identity
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LowRankProj(nn.Module):
    """Residual low-rank linear: x -> U(V x).

    U is initialized to zero so step-0 behavior is identity (h + P(h) = h).
    Pair with ``residual_apply`` for v1-style (I + P).

    Args:
        d: model hidden dim.
        r: low-rank inner dim (e.g. 64). Set r >= d for full-rank.
    """

    def __init__(self, d: int, r: int):
        super().__init__()
        self.U = nn.Linear(r, d, bias=False)
        self.V = nn.Linear(d, r, bias=False)
        nn.init.zeros_(self.U.weight)
        nn.init.normal_(self.V.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(self.V(x))


class FullProj(nn.Module):
    """Full d x d residual projection. Zero-init so step 0 = identity."""

    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.W.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W(x)


def make_projector(d: int, *, rank: int = 0) -> nn.Module:
    """Factory: rank=0 -> full d x d, otherwise low-rank with inner dim ``rank``."""
    return FullProj(d) if rank <= 0 else LowRankProj(d, rank)


def residual_apply(P: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute ``(I + P) x = x + P(x)``."""
    return x + P(x)
