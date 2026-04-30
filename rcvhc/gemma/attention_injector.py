"""Gemma-style Q/K/V attention-memory intervention."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from rcvhc.core.types import AttentionMemoryItem, QKVTrace
from rcvhc.gemma.model_adapter import get_decoder


QKV_INTERVENTION_MODES = {
    "no_memory",
    "delta_v",
    "delta_qv",
    "delta_kv",
    "delta_qkv",
    "delta_qv_zero",
    "delta_qv_random",
    "delta_qv_shuffled",
    "delta_qv_force_gate",
}


class UnsupportedGemmaStructure(ValueError):
    pass


class QKVDeltaProjector(nn.Module):
    """Project compressed Delta memory into q/k/v residual spaces."""

    def __init__(self, memory_dim: int, projection_dim: int, alpha_scale: float = 0.2, gate_bias: float = -1.0) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.projection_dim = int(projection_dim)
        self.alpha_scale = float(alpha_scale)
        self.norm = nn.LayerNorm(memory_dim)
        self.to_q = nn.Linear(memory_dim, projection_dim)
        self.to_k = nn.Linear(memory_dim, projection_dim)
        self.to_v = nn.Linear(memory_dim, projection_dim)
        self.lazy_proj = nn.ModuleDict()
        self.gate_q = nn.Linear(memory_dim, 1)
        self.gate_k = nn.Linear(memory_dim, 1)
        self.gate_v = nn.Linear(memory_dim, 1)
        nn.init.normal_(self.to_q.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.to_k.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.to_v.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.to_q.bias)
        nn.init.zeros_(self.to_k.bias)
        nn.init.zeros_(self.to_v.bias)
        for gate in (self.gate_q, self.gate_k, self.gate_v):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, gate_bias)

    def delta_vector(self, memories: list[AttentionMemoryItem], mode: str, kind: str, device, dtype) -> torch.Tensor:
        if mode in {"delta_qv_zero", "no_memory"} or not memories:
            return torch.zeros(1, 1, self.memory_dim, device=device, dtype=dtype)
        memories = list(memories)
        if mode == "delta_qv_shuffled" and len(memories) > 1:
            memories = list(reversed(memories))
        if kind == "q":
            values = [item.delta_q for item in memories]
        elif kind == "k":
            values = [item.delta_k for item in memories]
        else:
            values = [item.delta_v for item in memories]
        # P0 uses the highest-ranked retrieved block for the Q/K/V residual.
        # This keeps the block-alignment control meaningful: shuffling swaps the
        # retrieved block that supplies Delta instead of leaving a commutative
        # mean unchanged.
        z = values[0].to(device=device, dtype=dtype).view(1, 1, -1)
        if mode == "delta_qv_random":
            noise = torch.randn_like(z)
            z_norm = z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            n_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            z = noise * (z_norm / n_norm)
        return z

    def project(self, base: torch.Tensor, z: torch.Tensor, kind: str, mode: str) -> tuple[torch.Tensor, dict[str, float]]:
        z = self.norm(z.to(device=base.device, dtype=base.dtype))
        force_gate = mode == "delta_qv_force_gate"
        gate_q = torch.ones((*z.shape[:-1], 1), device=z.device, dtype=z.dtype) if force_gate else torch.sigmoid(self.gate_q(z))
        gate_k = torch.ones_like(gate_q) if force_gate else torch.sigmoid(self.gate_k(z))
        gate_v = torch.ones_like(gate_q) if force_gate else torch.sigmoid(self.gate_v(z))
        update = torch.zeros_like(base)
        if kind == "q" and mode in {"delta_qv", "delta_qkv", "delta_qv_shuffled", "delta_qv_zero", "delta_qv_random", "delta_qv_force_gate"}:
            update = self.alpha_scale * gate_q * self._project_kind("q", z, base.shape[-1], base.device, base.dtype)
        elif kind == "k" and mode in {"delta_kv", "delta_qkv"}:
            update = self.alpha_scale * gate_k * self._project_kind("k", z, base.shape[-1], base.device, base.dtype)
        elif kind == "v" and mode in QKV_INTERVENTION_MODES - {"no_memory"}:
            update = self.alpha_scale * gate_v * self._project_kind("v", z, base.shape[-1], base.device, base.dtype)
        trace = {
            f"{kind}_delta_norm": float(update.detach().float().norm().cpu()),
            f"{kind}_relative_delta_norm": float((update.detach().float().norm() / (base.detach().float().norm() + 1e-6)).cpu()),
            "gate_q": float(gate_q.detach().float().mean().cpu()),
            "gate_k": float(gate_k.detach().float().mean().cpu()),
            "gate_v": float(gate_v.detach().float().mean().cpu()),
        }
        return base + update.to(dtype=base.dtype), trace

    def _project_kind(self, kind: str, z: torch.Tensor, out_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.ensure_projection(kind, out_dim, device, dtype)
        default = {"q": self.to_q, "k": self.to_k, "v": self.to_v}[kind]
        if default.out_features == out_dim:
            return default(z)
        key = f"{kind}_{out_dim}"
        return self.lazy_proj[key](z)

    def ensure_projection(self, kind: str, out_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        default = {"q": self.to_q, "k": self.to_k, "v": self.to_v}[kind]
        if default.out_features == out_dim:
            default.to(device=device, dtype=dtype)
            return
        key = f"{kind}_{out_dim}"
        if key not in self.lazy_proj:
            layer = nn.Linear(self.memory_dim, out_dim)
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)
            self.lazy_proj[key] = layer
        self.lazy_proj[key].to(device=device, dtype=dtype)


@dataclass
class InjectionResult:
    logits: torch.Tensor
    trace: QKVTrace


class GemmaAttentionInjector:
    """Hook q_proj/k_proj/v_proj on a Gemma/Llama-style decoder layer."""

    def __init__(self, model: nn.Module, projector: QKVDeltaProjector) -> None:
        self.model = model
        self.projector = projector
        self.decoder = get_decoder(model)
        self.layers = getattr(self.decoder, "layers", None)
        if self.layers is None:
            raise UnsupportedGemmaStructure("expected Gemma/Llama-style model.model.layers")

    def materialize_for_layer(self, layer_id: int, device: torch.device, dtype: torch.dtype) -> None:
        layer = self.layers[layer_id]
        attn = getattr(layer, "self_attn", None)
        if not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            raise UnsupportedGemmaStructure("target layer does not expose q_proj/k_proj/v_proj")
        for kind, module_name in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj")):
            module = getattr(attn, module_name)
            out_dim = int(getattr(module, "out_features", 0))
            if out_dim <= 0:
                raise UnsupportedGemmaStructure(f"{module_name} does not expose out_features")
            self.projector.ensure_projection(kind, out_dim, device, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        layer_id: int,
        memories: list[AttentionMemoryItem],
        mode: str,
    ) -> InjectionResult:
        if mode not in QKV_INTERVENTION_MODES:
            raise ValueError(f"unsupported qkv mode: {mode}")
        if mode == "no_memory":
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=False, use_cache=False)
            return InjectionResult(out.logits, QKVTrace())
        layer = self.layers[layer_id]
        attn = getattr(layer, "self_attn", None)
        if not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            raise UnsupportedGemmaStructure("target layer does not expose q_proj/k_proj/v_proj")

        metrics: dict[str, float] = {}

        def hook_for(kind: str):
            def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
                z = self.projector.delta_vector(memories, mode=mode, kind=kind, device=output.device, dtype=output.dtype)
                updated, trace = self.projector.project(output, z, kind=kind, mode=mode)
                metrics.update(trace)
                return updated

            return hook

        handles = [
            attn.q_proj.register_forward_hook(hook_for("q")),
            attn.k_proj.register_forward_hook(hook_for("k")),
            attn.v_proj.register_forward_hook(hook_for("v")),
        ]
        try:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=False, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        trace = QKVTrace(
            q_delta_norm=metrics.get("q_delta_norm", 0.0),
            k_delta_norm=metrics.get("k_delta_norm", 0.0),
            v_delta_norm=metrics.get("v_delta_norm", 0.0),
            q_relative_delta_norm=metrics.get("q_relative_delta_norm", 0.0),
            k_relative_delta_norm=metrics.get("k_relative_delta_norm", 0.0),
            v_relative_delta_norm=metrics.get("v_relative_delta_norm", 0.0),
            gate_q=metrics.get("gate_q", 0.0),
            gate_k=metrics.get("gate_k", 0.0),
            gate_v=metrics.get("gate_v", 0.0),
        )
        return InjectionResult(out.logits, trace)
