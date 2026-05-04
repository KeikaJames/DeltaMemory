"""Dynamic LOPI (Layer-Orthogonal-Projection Injection) — Phase R, v3.3.

Architecture-agnostic, training-free injection wrapper for the attention-native
DeltaMemory bank.  Replaces the legacy

    out_bank = weights[..., T:] @ (alpha * mv_e)

at the merged-softmax branch of ``attn_native_bank.py`` with

    M_perp        = M_V - proj_{V_ctx}(M_V)
    out_bank_lopi = gamma_t * w(layer, t) * M_perp

so that:

* **Orthogonal Novelty** — only the component of the bank value that is
  perpendicular to the native context value is mixed into the residual
  stream.  Parallel (redundant) energy is dropped.
* **Adaptive Layer Routing** — a Gaussian window over layer index ℓ
  centred at μ_t (driven by the previous step's residual norm) and width
  σ_t (shrunk by the running mHC max-σ unstable-warning) modulates the
  per-layer injection strength.
* **Derivative Gate** — a Sigmoid on the per-step Q-derivative ‖Q_t − Q_{t-1}‖
  silences injection when the topic is stable and opens it during topic
  shifts.

All three components are independent and can be turned on/off via the
``LOPIConfig`` flags so the R-3 ablation grid (A0..A4) is a pure config sweep.

Shape convention
----------------
``V_ctx``, ``M_V``  : (B, H, T, D)            — post-softmax sub-readouts
``Q_t``, ``Q_prev`` : (B, H, T, D)            — pre/post-RoPE Q (configurable)
``prev_residual_norms`` : list[float] length L (or torch scalar tensor)

Bit-equal degeneracy
--------------------
With ``LOPIConfig.enabled = False`` (default) the module is **never** called
and the merged-softmax branch keeps the legacy formula.  With
``LOPIConfig(enabled=True, orthogonal=False, gaussian=False, derivative=False)``
the formula reduces to ``out_bank = alpha * (weights[..., T:] @ mv_e)`` which
is *also* bit-for-bit equivalent to the legacy code path (gamma=1, w=1,
M_perp = M_V).  Verified by ``tests/test_lopi_module.py``.

Author: KeikaJames, 2026-05-04 (Phase R freeze, PREREGISTRATION lopi_v33).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# Configuration


@dataclass
class LOPIConfig:
    """Frozen-by-preregistration hyperparameters for Dynamic LOPI v3.3."""

    enabled: bool = False  # master switch; False => module never invoked

    # Component switches (R-3 ablation grid A0..A4)
    orthogonal: bool = True     # M_perp = M_V - proj_{V_ctx}(M_V)
    gaussian: bool = True       # w(ell, t) = Gaussian over layer index
    derivative: bool = True     # gamma_t  = Sigmoid on ||Q_t - Q_{t-1}||

    # Derivative gate (Section 3.1 of PREREGISTRATION)
    k_gate: float = 5.0
    theta_gate: float = 0.5

    # Adaptive Layer Gaussian (Section 3.2)
    kappa_depth: float = 2.0
    beta_sigma: float = 2.0
    norm_base: float = 10.0          # default; calibrated per-model and overridden
    mu_low: float = 0.3              # mu in [mu_low * L, (mu_low + mu_span) * L]
    mu_span: float = 0.5
    sigma_floor: float = 1e-3        # numerical floor on Gaussian width

    # Numerical
    eps: float = 1e-6

    def asdict(self) -> dict:
        return {
            "enabled": self.enabled,
            "orthogonal": self.orthogonal,
            "gaussian": self.gaussian,
            "derivative": self.derivative,
            "k_gate": self.k_gate,
            "theta_gate": self.theta_gate,
            "kappa_depth": self.kappa_depth,
            "beta_sigma": self.beta_sigma,
            "norm_base": self.norm_base,
            "mu_low": self.mu_low,
            "mu_span": self.mu_span,
            "sigma_floor": self.sigma_floor,
            "eps": self.eps,
        }


# ---------------------------------------------------------------------------
# Per-call state — owned by the bank, mutated by the patched forward


@dataclass
class LOPIState:
    """Cross-step LOPI scratchpad held on the bank instance.

    All tensors are small (per-head Q vectors, per-layer scalar norms) so
    keeping them around for every decoded token is cheap.  The state is
    cleared by ``reset()`` when a new conversation / write-pass begins.
    """

    num_layers: int

    # Causality trick: caches from time-step t-1
    prev_q_per_layer: dict = field(default_factory=dict)         # layer_idx -> (B, H, T, D)
    prev_residual_norms: dict = field(default_factory=dict)      # layer_idx -> float
    mhc_sigma_max_running: float = 0.0
    mhc_sigma_count: int = 0

    def reset(self) -> None:
        self.prev_q_per_layer = {}
        self.prev_residual_norms = {}
        self.mhc_sigma_max_running = 0.0
        self.mhc_sigma_count = 0

    def update_mhc_sigma(self, sigma_max: float) -> None:
        # Running mean (cheap; avoids storing history).
        n = self.mhc_sigma_count
        self.mhc_sigma_max_running = (self.mhc_sigma_max_running * n + float(sigma_max)) / (n + 1)
        self.mhc_sigma_count = n + 1

    def avg_prev_residual_norm(self) -> float:
        if not self.prev_residual_norms:
            return 0.0
        vals = list(self.prev_residual_norms.values())
        return float(sum(vals) / len(vals))


# ---------------------------------------------------------------------------
# Pure-tensor primitives (test-friendly, no module state)


def derivative_gate(q_t: torch.Tensor, q_prev: Optional[torch.Tensor],
                    k: float, theta: float) -> torch.Tensor:
    """Compute gamma_t = sigmoid(k * (||Q_t - Q_{t-1}||_2 - theta)).

    Returns a tensor of shape (B, H, T, 1) so it broadcasts over D.  When
    ``q_prev is None`` (t = 0 or after reset) gamma_t collapses to 1 — this
    matches PREREGISTRATION §3.5 (legacy behavior at session boundaries).
    """
    if q_prev is None:
        # Use a tensor (not python 1.0) so autograd / dtype propagate cleanly.
        return torch.ones((*q_t.shape[:-1], 1), dtype=q_t.dtype, device=q_t.device)
    delta_q = torch.linalg.vector_norm(q_t - q_prev, ord=2, dim=-1, keepdim=True)
    # In bf16/fp16 the sigmoid argument range matters; cast to fp32 for the
    # transcendental, then back. Cheap on every backend.
    arg = (delta_q.float() - theta) * k
    gamma = torch.sigmoid(arg).to(q_t.dtype)
    return gamma


def layer_gaussian_weight(layer_idx: int, num_layers: int,
                          avg_prev_norm: float, mhc_sigma_max: float,
                          cfg: LOPIConfig,
                          device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute w(ell, t) for the current layer.

    Returns a 0-dim tensor that broadcasts against (B, H, T, 1).
    """
    L = float(num_layers)
    # depth_t in (0, 1)
    depth_arg = cfg.kappa_depth * (avg_prev_norm / max(cfg.norm_base, cfg.eps) - 1.0)
    depth_t = 1.0 / (1.0 + pow(2.718281828, -depth_arg))  # cheap fp scalar sigmoid
    mu_t = L * (cfg.mu_low + cfg.mu_span * depth_t)
    sigma_t = max((L / 6.0) * pow(2.718281828, -cfg.beta_sigma * float(mhc_sigma_max)),
                  cfg.sigma_floor)
    w = pow(2.718281828, -((float(layer_idx) - mu_t) ** 2) / (2.0 * sigma_t * sigma_t))
    return torch.tensor(w, device=device, dtype=dtype)


def orthogonal_novelty(m_v: torch.Tensor, v_ctx: torch.Tensor,
                       eps: float) -> torch.Tensor:
    """Return M_perp = M_V - proj_{V_ctx}(M_V).

    Reduction is along the last dim (head dimension D).  Output shape ==
    ``m_v.shape``.  When ``v_ctx`` is the zero vector the projection
    collapses to zero (epsilon guard) and ``M_perp == M_V``.
    """
    v_norm_sq = (v_ctx * v_ctx).sum(dim=-1, keepdim=True) + eps
    dot = (m_v * v_ctx).sum(dim=-1, keepdim=True)
    m_parallel = (dot / v_norm_sq) * v_ctx
    return m_v - m_parallel


# ---------------------------------------------------------------------------
# Top-level callable used inside the patched attention forward.


def apply_lopi(
    out_bank_native: torch.Tensor,   # legacy: weights[..., T:] @ (alpha * mv_e)
    v_ctx_readout: torch.Tensor,     # weights[..., :T] @ v_repeat (out_orig)
    q_post: torch.Tensor,            # current Q used for derivative gate
    layer_idx: int,
    state: LOPIState,
    cfg: LOPIConfig,
) -> torch.Tensor:
    """Apply Dynamic LOPI to one attention layer's bank readout.

    The function is **pure with respect to its config**: the only side
    effect on ``state`` is the t -> t+1 cache update at the end (so the
    caller does not have to remember to call a separate ``state.step()``).

    Returns the new ``out_bank`` tensor of identical shape to the input.
    Caller should add it to ``out_orig`` exactly as before.
    """
    if not cfg.enabled:
        return out_bank_native

    # 1. Orthogonal projection (op on the *readouts*, head-wise).
    if cfg.orthogonal:
        m_perp = orthogonal_novelty(out_bank_native, v_ctx_readout, cfg.eps)
    else:
        m_perp = out_bank_native

    # 2. Layer Gaussian (scalar, broadcasts).
    if cfg.gaussian:
        w_ell = layer_gaussian_weight(
            layer_idx=layer_idx,
            num_layers=state.num_layers,
            avg_prev_norm=state.avg_prev_residual_norm(),
            mhc_sigma_max=state.mhc_sigma_max_running,
            cfg=cfg,
            device=m_perp.device,
            dtype=m_perp.dtype,
        )
    else:
        w_ell = torch.tensor(1.0, device=m_perp.device, dtype=m_perp.dtype)

    # 3. Derivative gate (per-(B,H,T) tensor, broadcasts over D).
    if cfg.derivative:
        q_prev = state.prev_q_per_layer.get(layer_idx)
        gamma_t = derivative_gate(q_post, q_prev, cfg.k_gate, cfg.theta_gate)
    else:
        gamma_t = torch.tensor(1.0, device=m_perp.device, dtype=m_perp.dtype)

    out_bank_lopi = gamma_t * w_ell * m_perp

    # 4. Update t-1 caches for the *next* step.  We keep the latest Q per
    # layer; residual norms are updated by the bank caller (it has access
    # to the post-attention residual that we don't see here).
    state.prev_q_per_layer[layer_idx] = q_post.detach()

    return out_bank_lopi


__all__ = [
    "LOPIConfig",
    "LOPIState",
    "apply_lopi",
    "derivative_gate",
    "layer_gaussian_weight",
    "orthogonal_novelty",
]
