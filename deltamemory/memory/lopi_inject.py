"""ECOR — Energy-Conserving Orthogonal Rotation injection operator (Phase X.7).

Opt-in replacement for LOPI's additive injection ``V_out = V_ctx + s·M_⊥``.
The ECOR variant rotates V_ctx toward M_⊥ by angle θ, preserving ‖V_ctx‖:

    θ = (max_theta_frac · π) · tanh(k · s)         where s = γ_t · w · α
    V_rot = cos(θ)·V_ctx + sin(θ)·(M_⊥ · ‖V_ctx‖/‖M_⊥‖)

Five safeguards over the naïve ECOR formulation
------------------------------------------------
1. **max_theta cap** — θ ≤ max_theta_frac · π (default 60°) prevents V_ctx erasure.
2. **soft_blend slider** — blends additive and rotated outputs; 0 = pure additive (default).
3. **direction_eps fallback** — if ‖M_⊥‖/‖V_ctx‖ < eps, fall back to additive (avoids
   direction noise when the memory bank contribution is degenerate/near-zero).
4. **per_head flag** — rotation is applied inside each head's D_head subspace by
   operating on the last dimension (head dim) of the value tensor.
5. **adaptive k** — k can be overridden per-call (e.g., by a profiler sweep).

Bit-equal degeneracy (red-line guarantee)
-----------------------------------------
When ``ECORConfig.enabled=False`` (the default) **OR** ``cfg.soft_blend=0``, the
function returns EXACTLY ``V_ctx + s·M_perp`` via early-return, with zero ECOR
computation — preserving bit-for-bit equality with the legacy additive LOPI path.

Wiring note
-----------
``lopi.py::apply_lopi`` returns the *bank contribution* ``s·M_perp``, while
``lopi_inject`` returns the *full readout* ``V_ctx + s·M_perp`` (or its ECOR
equivalent). These interfaces differ by a ``V_ctx`` addend; wiring into
``apply_lopi`` would require restructuring the caller's summation site in
``attn_native_bank.py``. To avoid silent regression risk, ECOR is left as a
standalone function for W-T3.6 to wire externally (opt-in only).

Author: KeikaJames, 2026-05-04 (Phase X.7, PREREGISTRATION ecor_v10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ECORConfig:
    """Opt-in ECOR hyperparameters.

    All defaults reproduce the legacy additive LOPI behaviour exactly:
    ``enabled=False`` and ``soft_blend=0`` both trigger an early-return to the
    plain ``V_ctx + s·M_perp`` formula.
    """

    enabled: bool = False              # opt-in; default OFF preserves existing additive LOPI
    soft_blend: float = 0.0            # 0=pure additive, 1=pure ECOR; sweep in W-T3.6
    k: Optional[float] = None          # if None, use 1.0 (safe); profiler may override
    per_head: bool = True              # rotate inside each head's D_head subspace
    direction_eps: float = 1e-3        # if ‖M_⊥‖/‖V_ctx‖ < eps, fallback to additive
    max_theta_frac: float = 1.0 / 3.0  # cap θ at max_theta_frac · π (1/3 ⇒ 60°)


# ---------------------------------------------------------------------------
# Internal broadcast helper
# ---------------------------------------------------------------------------


def _align_gamma(gamma_t: torch.Tensor, V_ctx: torch.Tensor) -> torch.Tensor:
    """Reshape gamma_t to be broadcastable with V_ctx (and therefore M_perp).

    Handles common shape combinations:

    * scalar (0-dim)            → unchanged
    * (B, T) with V_ctx (B, T, D)        → (B, T, 1)
    * (B, T) with V_ctx (B, H, T, D)     → (B, 1, T, 1)
    * (B, H, T) with V_ctx (B, H, T, D)  → (B, H, T, 1)
    * (B, H, T, 1) with V_ctx (B, H, T, D) → unchanged (already correct)
    """
    if gamma_t.dim() == 0:
        return gamma_t

    v_ndim = V_ctx.dim()
    g_ndim = gamma_t.dim()

    if g_ndim >= v_ndim:
        return gamma_t  # already at least as many dims as V_ctx

    if v_ndim == 3 and g_ndim == 2:
        return gamma_t.unsqueeze(-1)              # (B, T) → (B, T, 1)

    if v_ndim == 4 and g_ndim == 2:
        return gamma_t.unsqueeze(1).unsqueeze(-1)  # (B, T) → (B, 1, T, 1)

    if v_ndim == 4 and g_ndim == 3:
        return gamma_t.unsqueeze(-1)              # (B, H, T) → (B, H, T, 1)

    # Generic fallback: append trailing singleton dims
    for _ in range(v_ndim - g_ndim):
        gamma_t = gamma_t.unsqueeze(-1)
    return gamma_t


# ---------------------------------------------------------------------------
# Main injection operator
# ---------------------------------------------------------------------------


def lopi_inject(
    V_ctx: torch.Tensor,         # (..., D) or (..., H, D_head)
    M_perp: torch.Tensor,        # same shape as V_ctx
    gamma_t: torch.Tensor,       # (...,) broadcastable to V_ctx less last dim(s)
    w_ell: torch.Tensor,         # scalar or layer-broadcast
    alpha_base: float = 1.0,
    cfg: Optional[ECORConfig] = None,
) -> torch.Tensor:
    """LOPI injection operator with optional ECOR rotation.

    When ``cfg.enabled=False`` (default) **or** ``cfg.soft_blend=0``, returns
    the EXACT additive baseline ``V_ctx + s · M_perp`` where
    ``s = gamma_t · w_ell · alpha_base`` — bit-equal to legacy LOPI.

    Parameters
    ----------
    V_ctx:
        Context value readout; shape (..., D) or (..., H, D_head).
    M_perp:
        Orthogonal bank component; same shape as V_ctx.
    gamma_t:
        Derivative gate scalar(s); broadcastable to V_ctx (minus last dim).
        Common inputs: scalar tensor, (B, T), (B, H, T, 1).
    w_ell:
        Layer Gaussian weight; scalar tensor.
    alpha_base:
        Global injection scale (Python float).
    cfg:
        :class:`ECORConfig` instance; ``None`` ⇒ use defaults (additive path).

    Returns
    -------
    torch.Tensor
        Same shape as V_ctx.
    """
    cfg = cfg or ECORConfig()

    # Align gamma_t shape so s broadcasts against M_perp/V_ctx
    gamma_bc = _align_gamma(gamma_t, V_ctx)
    s = gamma_bc * w_ell * alpha_base                         # additive scalar(s)

    # -----------------------------------------------------------------------
    # ADDITIVE PATH — bit-equal early return
    # -----------------------------------------------------------------------
    V_add = V_ctx + s * M_perp

    if not cfg.enabled or cfg.soft_blend == 0.0:
        return V_add

    # -----------------------------------------------------------------------
    # ECOR ROTATION PATH (opt-in)
    # -----------------------------------------------------------------------
    reduce_dim = -1  # operate on the head/feature dimension

    norm_v = V_ctx.norm(dim=reduce_dim, keepdim=True).clamp_min(1e-8)
    norm_m = M_perp.norm(dim=reduce_dim, keepdim=True)

    # Safeguard 3: direction stability gate
    direction_stable = (norm_m / norm_v) > cfg.direction_eps

    k = cfg.k if cfg.k is not None else 1.0

    # θ lives in the same broadcast shape as s (without the trailing D-dim).
    # s may have shape (..., 1) or be a 0-dim scalar tensor.
    theta = (cfg.max_theta_frac * math.pi) * torch.tanh(k * s)
    # ↑ theta has same ndim as s; cos/sin will broadcast over V_ctx's last dim
    # via the explicit unsqueeze below when needed.

    # Safeguard 1: max_theta_frac ensures θ ≤ max_theta_frac·π (e.g. 60°)
    # — already embedded in the formula above via the scalar prefix.

    # Safeguard 4 (per_head): the norm reduction on dim=-1 naturally operates
    # on each head's D_head subspace when V_ctx is (B, H, T, D_head).

    # Scale M_perp to have the same norm as V_ctx (making rotation isometric)
    M_unit_scaled = M_perp * (norm_v / norm_m.clamp_min(1e-8))

    # cos/sin: unsqueeze last dim if theta has fewer dims than V_ctx
    if theta.dim() < V_ctx.dim():
        cos_t = torch.cos(theta).unsqueeze(reduce_dim)
        sin_t = torch.sin(theta).unsqueeze(reduce_dim)
    else:
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

    V_rot = cos_t * V_ctx + sin_t * M_unit_scaled

    # Safeguard 2: soft blend between additive and rotated
    V_blend = (1.0 - cfg.soft_blend) * V_add + cfg.soft_blend * V_rot

    # Safeguard 3 (cont.): where direction is unstable, fall back to additive
    return torch.where(direction_stable, V_blend, V_add)


__all__ = [
    "ECORConfig",
    "lopi_inject",
]
