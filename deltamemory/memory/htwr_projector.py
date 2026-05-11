"""InfoNCE-trained key projector for HTWR T4 (Exp12).

Projects per-layer residual keys/queries through a tiny linear or low-rank/MLP
head so that ``cos(φ(q), φ(k_correct)) >> cos(φ(q), φ(k_other))``.

Trained on CPU (or MPS) in fp32.  The projector is shared across layers by
default; we score by max-over-layers of cosine in the projected space.

Usage:

    proj = LinearProjector(d_in=2560, d_out=64)
    train_projector(proj, pos_pairs, neg_bank, epochs=200, lr=1e-3, tau=0.07)
    # Then plug into a custom retriever (not provided here — see exp12 runner).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

ProjectorKind = Literal["linear", "lowrank", "mlp", "shared"]


class LinearProjector(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=bias)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LowRankResidualProjector(nn.Module):
    def __init__(self, d_in: int, rank: int) -> None:
        super().__init__()
        self.U = nn.Linear(d_in, rank, bias=False)
        self.V = nn.Linear(rank, d_in, bias=False)
        nn.init.zeros_(self.V.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.V(self.U(x))


class MLPProjector(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def make_projector(
    kind: ProjectorKind,
    *,
    d_in: int,
    d_out: int = 64,
    rank: int = 32,
    d_hidden: int = 128,
) -> nn.Module:
    if kind in ("linear", "shared"):
        return LinearProjector(d_in, d_out)
    if kind == "lowrank":
        return LowRankResidualProjector(d_in, rank)
    if kind == "mlp":
        return MLPProjector(d_in, d_hidden, d_out)
    raise ValueError(f"Unknown projector kind: {kind}")


@dataclass
class InfoNCEConfig:
    tau: float = 0.07
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32


def info_nce_loss(
    q_proj: torch.Tensor,  # (B, K)
    k_proj: torch.Tensor,  # (M, K) — positives + negatives, with positive at row pos_idx[b]
    pos_idx: torch.Tensor,  # (B,)
    tau: float,
) -> torch.Tensor:
    qn = F.normalize(q_proj, dim=-1)
    kn = F.normalize(k_proj, dim=-1)
    logits = (qn @ kn.T) / tau           # (B, M)
    return F.cross_entropy(logits, pos_idx.long())


def train_projector(
    proj: nn.Module,
    queries: torch.Tensor,        # (N, d_in)
    keys: torch.Tensor,           # (N, d_in) — keys[i] is the positive for queries[i]
    *,
    config: InfoNCEConfig | None = None,
    verbose: bool = False,
) -> list[float]:
    """In-batch InfoNCE training. Each batch uses all other keys as negatives."""
    cfg = config or InfoNCEConfig()
    opt = torch.optim.Adam(proj.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = queries.shape[0]
    losses: list[float] = []
    proj.train()
    for ep in range(cfg.epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        batches = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            q = queries[idx]
            k = keys[idx]
            q_p = proj(q)
            k_p = proj(k)
            pos = torch.arange(q.shape[0])
            loss = info_nce_loss(q_p, k_p, pos, cfg.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            batches += 1
        losses.append(ep_loss / max(1, batches))
        if verbose and (ep == 0 or (ep + 1) % max(1, cfg.epochs // 10) == 0):
            print(f"[infonce] epoch {ep+1}/{cfg.epochs} loss={losses[-1]:.4f}")
    proj.eval()
    return losses


__all__ = [
    "ProjectorKind",
    "LinearProjector",
    "LowRankResidualProjector",
    "MLPProjector",
    "make_projector",
    "InfoNCEConfig",
    "info_nce_loss",
    "train_projector",
]
