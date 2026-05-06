"""Novelty and importance helpers for memory-bank rows."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _flatten_rows(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 2:
        return t.reshape(1, -1).float()
    return t.reshape(t.size(0), -1).float()


def _existing_features(bank_state: dict[str, Any]) -> torch.Tensor:
    parts = []
    for k in bank_state.get("M_K", []):
        if k.size(0) > 0:
            parts.append(_flatten_rows(k.detach()).cpu())
    if not parts:
        return torch.empty(0, 0)
    return F.normalize(torch.cat(parts, dim=1), dim=1, eps=1e-12)


def compute_novelty(K_new: torch.Tensor, bank_state: dict[str, Any]) -> float:
    """Return ``1 - max(cos(K_new, K_existing))``; empty banks return 1.0."""
    with torch.no_grad():
        existing = _existing_features(bank_state)
        if existing.numel() == 0:
            return 1.0
        k = F.normalize(_flatten_rows(K_new.detach()).cpu(), dim=1, eps=1e-12)
        if k.size(0) != 1:
            k = F.normalize(k.mean(dim=0, keepdim=True), dim=1, eps=1e-12)
        max_cos = torch.matmul(k, existing.T).max().item()
        return float(max(0.0, min(2.0, 1.0 - max_cos)))


def importance_bias(bank_state: dict[str, Any]) -> torch.Tensor:
    """Return a multiplicative attention-score bias of shape ``[bank_len]``.

    Explicit ``importance_scores`` are mapped to ``1 + score``. If absent,
    ``merge_counts`` provide a small utility boost via ``1 + log1p(count)/max``.
    """
    with torch.no_grad():
        m_k = bank_state.get("M_K") or []
        n = int(m_k[0].size(0)) if m_k else 0
        if n == 0:
            return torch.empty(0, dtype=torch.float32)
        scores = bank_state.get("importance_scores")
        if scores is not None and len(scores) == n:
            s = torch.as_tensor(scores, dtype=torch.float32).flatten().clamp_min(0.0)
            return 1.0 + s
        counts = bank_state.get("merge_counts")
        if counts is not None and len(counts) == n:
            c = torch.as_tensor(counts, dtype=torch.float32).flatten().clamp_min(1.0)
            util = torch.log1p(c)
            return 1.0 + util / util.max().clamp_min(1e-12)
        return torch.ones(n, dtype=torch.float32)
