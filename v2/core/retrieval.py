"""TopK retrieval over AttentionBank — alternative to all-attend.

The default LPL bank read attends over ALL bank entries, scaling O(N_b * T)
in compute. This module provides cheap TopK selectors so a bank with N_b=10K
entries can be queried in O(K * T) at attention time.

Three modes:
    - ``cosine``: cosine similarity between query hidden and bank h-vector
    - ``dot``: raw dot product
    - ``learned``: a small linear address head maps query -> address logits

Returned indices can be applied to ``bank.slots[layer]`` BEFORE the K/V
projection in the attention patch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_cosine(query: torch.Tensor, bank: torch.Tensor, k: int) -> torch.Tensor:
    """query [..., d], bank [N, d] -> indices [..., k] of top-k by cosine."""
    q = F.normalize(query.float(), dim=-1)
    b = F.normalize(bank.float(), dim=-1)
    scores = q @ b.t()
    return scores.topk(min(k, b.shape[0]), dim=-1).indices


def topk_dot(query: torch.Tensor, bank: torch.Tensor, k: int) -> torch.Tensor:
    scores = query.float() @ bank.float().t()
    return scores.topk(min(k, bank.shape[0]), dim=-1).indices


class LearnedAddress(nn.Module):
    """Small linear address head: q -> N_b logits (uses fixed bank as keys)."""

    def __init__(self, d: int):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, query: torch.Tensor, bank: torch.Tensor, k: int) -> torch.Tensor:
        scores = self.proj(query.float()) @ bank.float().t()
        return scores.topk(min(k, bank.shape[0]), dim=-1).indices


def gather_bank_topk(bank_layer: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """bank_layer [N, d], indices [k] -> [k, d]."""
    return bank_layer[indices]
