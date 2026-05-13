"""Temporary fast-weight readouts for Mneme payloads."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class LMHeadFastWeightProjector(nn.Module):
    """Generate a temporary low-rank LM-head update from one memory payload.

    The frozen LM head is never mutated. The generated update is applied only to
    the current forward pass:

        logits' = logits + scale * sum_r (hidden @ u_r) * (W_out @ v_r)
    """

    def __init__(
        self,
        memory_dim: int,
        hidden_size: int,
        output_embeddings: nn.Module,
        *,
        rank: int = 1,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.scale = float(scale)
        self.norm = nn.LayerNorm(memory_dim)
        self.to_u = nn.Linear(memory_dim, self.rank * hidden_size)
        self.to_v = nn.Linear(memory_dim, self.rank * hidden_size)
        self.output_embeddings = output_embeddings
        for param in self.output_embeddings.parameters():
            param.requires_grad_(False)
        nn.init.normal_(self.to_u.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.to_v.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.to_u.bias)
        nn.init.zeros_(self.to_v.bias)

    def forward(self, base_logits: torch.Tensor, hidden: torch.Tensor, payload: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        z = self.norm(payload.to(device=hidden.device, dtype=hidden.dtype))
        u = self.to_u(z).view(self.rank, self.hidden_size)
        v = self.to_v(z).view(self.rank, self.hidden_size)
        u = F.normalize(u.float(), dim=-1)
        v = F.normalize(v.float(), dim=-1)
        activation = torch.einsum("bsh,rh->bsr", hidden.float(), u) / math.sqrt(self.hidden_size)
        vocab_direction = torch.einsum("vh,rh->rv", self.output_embeddings.weight.float(), v)
        update = torch.einsum("bsr,rv->bsv", activation, vocab_direction)
        update = (self.scale * update).to(device=base_logits.device, dtype=base_logits.dtype)
        logits = base_logits + update
        return logits, {
            "lm_head_lora_rank": float(self.rank),
            "lm_head_lora_update_norm": float(update.detach().float().norm().cpu()),
            "lm_head_lora_u_norm": float(u.detach().float().norm().cpu()),
            "lm_head_lora_v_norm": float(v.detach().float().norm().cpu()),
        }
