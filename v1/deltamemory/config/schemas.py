"""Pydantic v2 config schemas for Mneme injector families and the FastAPI service runtime.

These schemas are schema-only — no existing injector is modified. Migration is a separate PR.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class BaseInjectorConfig(BaseModel):
    """Shared fields for all injector configs."""

    alpha: float = Field(..., ge=0, description="Injection strength; must be ≥ 0.")
    layer_idx: int = Field(-1, ge=-1, description="Target layer index; -1 = auto-select.")
    enabled: bool = Field(True, description="Whether this injector is active.")
    dtype: Literal["fp32", "bf16", "fp16"] = Field("bf16", description="Compute dtype.")
    device: Literal["cpu", "cuda", "mps"] = Field("cpu", description="Target device.")


class LopiConfig(BaseInjectorConfig):
    """Config for the LOPI (Latent-Offset Projection Injection) injector."""

    eta_sigma: float = Field(
        ...,
        gt=0,
        le=2,
        description="Noise scale σ; must be in (0, 2].",
    )
    gate_k: float = Field(..., gt=0, description="Sigmoid gate steepness; must be > 0.")
    gate_theta: float = Field(..., ge=0, description="Sigmoid gate shift; must be ≥ 0.")
    use_derivative_gate: bool = Field(True, description="Enable derivative-based gating.")


class CaaConfig(BaseInjectorConfig):
    """Config for the CAA (Contrastive Activation Addition) injector."""

    n_pairs: int = Field(..., ge=1, description="Number of contrastive pairs; must be ≥ 1.")
    use_lopi_gate: bool = Field(False, description="Route through LOPI gate when True.")
    gate_k: float = Field(..., gt=0, description="Sigmoid gate steepness; must be > 0.")
    gate_theta: float = Field(..., ge=0, description="Sigmoid gate shift; must be ≥ 0.")


class ScarConfig(BaseInjectorConfig):
    """Config for the SCAR (Subspace Constraint and Retention) injector."""

    projection: Literal["m_perp", "raw"] = Field(
        "m_perp",
        description="Projection mode: 'm_perp' = orthogonal complement, 'raw' = no projection.",
    )
    subspace_rank: int = Field(..., ge=1, description="Rank of the retained subspace; must be ≥ 1.")


class AttnNativeBankConfig(BaseInjectorConfig):
    """Config for the Attention Native Bank injector."""

    bank_size: int = Field(..., ge=1, description="Maximum number of stored attention vectors.")
    top_k: int = Field(..., ge=1, description="Number of top matches to retrieve; must be ≤ bank_size.")
    theta: float = Field(
        ...,
        description="Angular threshold in radians; must be in [-π, π].",
    )
    capture: Literal["pre_rope", "post_rope"] = Field(
        "pre_rope",
        description="Whether to capture keys before or after RoPE application.",
    )

    @field_validator("theta")
    @classmethod
    def theta_in_range(cls, v: float) -> float:
        if not (-math.pi <= v <= math.pi):
            raise ValueError(f"theta must be in [-π, π], got {v}")
        return v

    @model_validator(mode="after")
    def top_k_le_bank_size(self) -> "AttnNativeBankConfig":
        if self.top_k > self.bank_size:
            raise ValueError(
                f"top_k ({self.top_k}) must not exceed bank_size ({self.bank_size})"
            )
        return self


class RomeWriterConfig(BaseInjectorConfig):
    """Config for the ROME (Rank-One Model Editing) writer injector."""

    n_optim_steps: int = Field(
        ..., ge=1, description="Number of optimisation steps; must be ≥ 1."
    )


class MnemeWriterConfig(BaseInjectorConfig):
    """Config for the Mneme memory-writer injector."""

    write_alpha: float = Field(
        ..., ge=0, description="Write-path scaling factor; must be ≥ 0."
    )


class ServiceConfig(BaseModel):
    """Runtime config for the FastAPI service."""

    bind_host: str = Field(..., description="Host address to bind the server to.")
    port: int = Field(..., ge=1, le=65535, description="TCP port; must be in [1, 65535].")
    workers: int = Field(..., ge=1, description="Number of Uvicorn workers; must be ≥ 1.")
    request_size_cap_mb: int = Field(
        ..., ge=1, description="Maximum allowed request body size in MiB; must be ≥ 1."
    )
    jwt_required: bool = Field(True, description="Require a valid JWT on every request.")
    prometheus_enabled: bool = Field(True, description="Expose /metrics endpoint.")
