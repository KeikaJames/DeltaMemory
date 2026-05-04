"""Gemma-style layerwise Q/K/V Delta Memory intervention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from deltamemory.core.types import AttentionMemoryItem, QKVTrace
from deltamemory.gemma.model_adapter import get_decoder

QKV_INTERVENTION_MODES = {
    "no_memory",
    "delta_v",
    "delta_qv",
    "delta_kv",
    "delta_qkv",
    "delta_qv_zero",
    "delta_qv_random",
    "delta_qv_shuffled",
    "delta_qv_wrong_layer",
    "delta_qv_wrong_query",
    "delta_qv_identity_gate",
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

    def identity_gate(self, memories: list[AttentionMemoryItem], mode: str, device, dtype) -> torch.Tensor:
        if mode != "delta_qv_identity_gate":
            return torch.ones(1, 1, 1, device=device, dtype=dtype)
        if not memories:
            return torch.zeros(1, 1, 1, device=device, dtype=dtype)
        value = float(memories[0].metadata.get("identity_gate", 0.0))
        return torch.full((1, 1, 1), value, device=device, dtype=dtype)

    def project(
        self,
        base: torch.Tensor,
        z: torch.Tensor,
        kind: str,
        mode: str,
        identity_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        z = self.norm(z.to(device=base.device, dtype=base.dtype))
        force_gate = mode == "delta_qv_force_gate"
        identity_gate = (
            torch.ones((*z.shape[:-1], 1), device=z.device, dtype=z.dtype)
            if identity_gate is None
            else identity_gate.to(device=z.device, dtype=z.dtype)
        )
        gate_q = torch.ones((*z.shape[:-1], 1), device=z.device, dtype=z.dtype) if force_gate else torch.sigmoid(self.gate_q(z))
        gate_k = torch.ones_like(gate_q) if force_gate else torch.sigmoid(self.gate_k(z))
        gate_v = torch.ones_like(gate_q) if force_gate else torch.sigmoid(self.gate_v(z))
        update = torch.zeros_like(base)
        if kind == "q" and mode in {"delta_qv", "delta_qkv", "delta_qv_shuffled", "delta_qv_zero", "delta_qv_random", "delta_qv_wrong_layer", "delta_qv_wrong_query", "delta_qv_identity_gate", "delta_qv_force_gate"}:
            update = identity_gate * self.alpha_scale * gate_q * self._project_kind("q", z, base.shape[-1], base.device, base.dtype)
        elif kind == "k" and mode in {"delta_kv", "delta_qkv"}:
            update = self.alpha_scale * gate_k * self._project_kind("k", z, base.shape[-1], base.device, base.dtype)
        elif kind == "v" and mode in QKV_INTERVENTION_MODES - {"no_memory"}:
            update = identity_gate * self.alpha_scale * gate_v * self._project_kind("v", z, base.shape[-1], base.device, base.dtype)
        trace = {
            f"{kind}_delta_norm": float(update.detach().float().norm().cpu()),
            f"{kind}_relative_delta_norm": float((update.detach().float().norm() / (base.detach().float().norm() + 1e-6)).cpu()),
            "gate_q": float(gate_q.detach().float().mean().cpu()),
            "gate_k": float(gate_k.detach().float().mean().cpu()),
            "gate_v": float(gate_v.detach().float().mean().cpu()),
            "identity_gate": float(identity_gate.detach().float().mean().cpu()),
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
    """Hook q_proj/k_proj/v_proj on Gemma/Llama-style decoder attention layers."""

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

    def materialize_for_layers(self, layer_ids: list[int], device: torch.device, dtype: torch.dtype) -> None:
        for layer_id in layer_ids:
            self.materialize_for_layer(layer_id, device, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        layer_id: int,
        memories: list[AttentionMemoryItem],
        mode: str,
    ) -> InjectionResult:
        return self.forward_layers(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memories_by_layer={int(layer_id): memories},
            mode=mode,
        )

    def forward_layers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        memories_by_layer: dict[int, list[AttentionMemoryItem]],
        mode: str,
    ) -> InjectionResult:
        if mode not in QKV_INTERVENTION_MODES:
            raise ValueError(f"unsupported qkv mode: {mode}")
        if mode == "no_memory":
            with torch.no_grad():
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=False, use_cache=False)
            return InjectionResult(out.logits, QKVTrace())

        memories_by_layer = _wrong_layer_memories(memories_by_layer) if mode == "delta_qv_wrong_layer" else memories_by_layer
        layer_ids = sorted(int(layer_id) for layer_id in memories_by_layer)
        metrics: dict[str, list[float]] = {}
        handles = []

        def add_metric(trace: dict[str, float]) -> None:
            for key, value in trace.items():
                metrics.setdefault(key, []).append(float(value))

        def hook_for(layer_memories: list[AttentionMemoryItem], kind: str):
            def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
                z = self.projector.delta_vector(layer_memories, mode=mode, kind=kind, device=output.device, dtype=output.dtype)
                identity_gate = self.projector.identity_gate(layer_memories, mode=mode, device=output.device, dtype=output.dtype)
                updated, trace = self.projector.project(output, z, kind=kind, mode=mode, identity_gate=identity_gate)
                add_metric(trace)
                return updated

            return hook

        for layer_id in layer_ids:
            layer = self.layers[layer_id]
            attn = getattr(layer, "self_attn", None)
            if not all(hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")):
                raise UnsupportedGemmaStructure("target layer does not expose q_proj/k_proj/v_proj")
            layer_memories = memories_by_layer[layer_id]
            handles.extend(
                [
                    attn.q_proj.register_forward_hook(hook_for(layer_memories, "q")),
                    attn.k_proj.register_forward_hook(hook_for(layer_memories, "k")),
                    attn.v_proj.register_forward_hook(hook_for(layer_memories, "v")),
                ]
            )
        try:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=False, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        def total(key: str) -> float:
            return sum(metrics.get(key, []))

        def mean(key: str) -> float:
            values = metrics.get(key, [])
            return sum(values) / len(values) if values else 0.0

        trace = QKVTrace(
            q_delta_norm=total("q_delta_norm"),
            k_delta_norm=total("k_delta_norm"),
            v_delta_norm=total("v_delta_norm"),
            q_relative_delta_norm=mean("q_relative_delta_norm"),
            k_relative_delta_norm=mean("k_relative_delta_norm"),
            v_relative_delta_norm=mean("v_relative_delta_norm"),
            gate_q=mean("gate_q"),
            gate_k=mean("gate_k"),
            gate_v=mean("gate_v"),
            identity_gate=mean("identity_gate"),
            injected_layers=float(len(layer_ids)),
        )
        return InjectionResult(out.logits, trace)


def _wrong_layer_memories(memories_by_layer: dict[int, list[AttentionMemoryItem]]) -> dict[int, list[AttentionMemoryItem]]:
    layer_ids = sorted(memories_by_layer)
    if len(layer_ids) < 2:
        return memories_by_layer
    rotated = layer_ids[1:] + layer_ids[:1]
    return {layer_id: memories_by_layer[source_layer] for layer_id, source_layer in zip(layer_ids, rotated)}
