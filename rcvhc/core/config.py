"""Configuration for Delta Memory experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


QKV_MODES = {
    "no_memory",
    "raw_memory",
    "hidden_retrieval",
    "delta_v",
    "delta_qv",
    "delta_kv",
    "delta_qkv",
    "delta_qv_zero",
    "delta_qv_random",
    "delta_qv_shuffled",
    "delta_qv_wrong_layer",
    "delta_qv_wrong_query",
    "delta_qv_force_gate",
}


@dataclass
class RCVHCCleanConfig:
    model_name: str = "mock-gemma"
    memory_dim: int = 128
    block_size: int = 128
    local_window: int = 1024
    top_k: int = 4
    layers: str = "all"
    enabled_layers: list[int] = field(default_factory=list)
    alpha_scale: float = 0.2
    gate_bias: float = -1.0
    dtype: str = "float32"
    device: str = "cpu"
    main_modes: tuple[str, ...] = (
        "no_memory",
        "raw_memory",
        "hidden_retrieval",
        "delta_qv",
        "delta_qv_zero",
        "delta_qv_random",
        "delta_qv_shuffled",
        "delta_qv_wrong_layer",
        "delta_qv_wrong_query",
        "delta_qv_force_gate",
    )

    def validate(self) -> None:
        if self.memory_dim <= 0:
            raise ValueError("memory_dim must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.local_window <= 0:
            raise ValueError("local_window must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        unknown = [mode for mode in self.main_modes if mode not in QKV_MODES]
        if unknown:
            raise ValueError(f"unsupported modes: {unknown}")


def resolve_layer_policy(policy: str, exposed_layers: Sequence[int]) -> list[int]:
    """Resolve a CLI layer policy into concrete exposed layer ids."""

    if not exposed_layers:
        raise ValueError("no exposed QKV layers were found")
    if policy == "max_exposed":
        return [int(max(exposed_layers))]
    if policy == "all":
        return [int(layer) for layer in exposed_layers]
    result: list[int] = []
    for raw in policy.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw.replace(".", "", 1).isdigit() and "." in raw:
            ratio = float(raw)
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"layer ratio must be in [0,1]: {raw}")
            target = int(round(ratio * (max(exposed_layers))))
            result.append(min(exposed_layers, key=lambda layer: abs(layer - target)))
        else:
            layer = int(raw)
            result.append(min(exposed_layers, key=lambda item: abs(item - layer)))
    if not result:
        raise ValueError(f"empty layer policy: {policy}")
    return sorted(set(result))
