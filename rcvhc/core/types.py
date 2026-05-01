"""Typed records shared by the cleanroom modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class AttentionMemoryItem:
    memory_id: int | None
    layer_id: int
    block_id: int
    token_start: int
    token_end: int
    raw_key: torch.Tensor
    raw_value: torch.Tensor
    delta_q: torch.Tensor
    delta_k: torch.Tensor
    delta_v: torch.Tensor
    usage_mass: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_cpu_detached(self) -> "AttentionMemoryItem":
        return AttentionMemoryItem(
            memory_id=self.memory_id,
            layer_id=self.layer_id,
            block_id=self.block_id,
            token_start=self.token_start,
            token_end=self.token_end,
            raw_key=self.raw_key.detach().cpu().float(),
            raw_value=self.raw_value.detach().cpu().float(),
            delta_q=self.delta_q.detach().cpu().float(),
            delta_k=self.delta_k.detach().cpu().float(),
            delta_v=self.delta_v.detach().cpu().float(),
            usage_mass=float(self.usage_mass),
            metadata=dict(self.metadata),
        )


@dataclass
class RetrievalRecord:
    memory_id: int
    score: float
    item: AttentionMemoryItem


@dataclass
class QKVTrace:
    q_delta_norm: float = 0.0
    k_delta_norm: float = 0.0
    v_delta_norm: float = 0.0
    q_relative_delta_norm: float = 0.0
    k_relative_delta_norm: float = 0.0
    v_relative_delta_norm: float = 0.0
    gate_q: float = 0.0
    gate_k: float = 0.0
    gate_v: float = 0.0
    injected_layers: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "q_delta_norm": self.q_delta_norm,
            "k_delta_norm": self.k_delta_norm,
            "v_delta_norm": self.v_delta_norm,
            "q_relative_delta_norm": self.q_relative_delta_norm,
            "k_relative_delta_norm": self.k_relative_delta_norm,
            "v_relative_delta_norm": self.v_relative_delta_norm,
            "gate_q": self.gate_q,
            "gate_k": self.gate_k,
            "gate_v": self.gate_v,
            "injected_layers": self.injected_layers,
        }
