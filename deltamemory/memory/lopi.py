"""Dynamic LOPI (Layer-Orthogonal-Projection Injection) — Phase R, v3.3.

Architecture-agnostic, training-free injection wrapper for the attention-native
Mneme bank.  Replaces the legacy

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
from typing import Any, List, Literal, Optional

import torch

# Forward-declared for typing; real import is lazy in case the profiler
# module is not loaded (legacy v3.4 paths still work).
try:  # pragma: no cover
    from deltamemory.memory.lopi_profiler import LOPIProfile  # noqa: F401
except Exception:  # pragma: no cover
    LOPIProfile = Any  # type: ignore[misc, assignment]


# ---------------------------------------------------------------------------
# Configuration


@dataclass
class LOPIConfig:
    """Frozen-by-preregistration hyperparameters for Dynamic LOPI v3.3."""

    enabled: bool = False  # master switch; False => module never invoked

    # Component switches (R-3 ablation grid A0..A4).
    # ----------------------------------------------------------------------
    # v3.4 (post-R-3 strike-1 update, 2026-05-04): the empirical 630-cell
    # ablation in `reports/cleanroom/lopi_v33/FINDINGS.md` showed that the
    # orthogonal-novelty projection M_perp INCREASES neutral-text drift in
    # the operating regime alpha in [0.25, 2] across all 6 GPT-2 cells,
    # while pure Gaussian focusing alone (variant A4) collapses
    # catastrophic drift at alpha=8 by 65-95%. We therefore flip the
    # default for `orthogonal` to False; the field is retained for
    # ablation runs that explicitly request the v3.3 behaviour.
    # ----------------------------------------------------------------------
    orthogonal: bool = False    # M_perp = M_V - proj_{V_ctx}(M_V)  (v3.3 default was True)
    gaussian: bool = True       # w(ell, t) = Gaussian over layer index
    derivative: bool = True     # gamma_t  = Sigmoid on ||Q_t - Q_{t-1}||

    # Profile mode (Phase S, v3.5):
    # ----------------------------------------------------------------------
    # ``"static"`` keeps the v3.4 hard-coded ``norm_base = 10.0`` global
    # constant so R-1..R-4 published numbers can be bit-equal-reproduced.
    # ``"auto"`` (default) reads ``LOPIState.profile`` (a per-architecture
    # ``LOPIProfile`` from ``lopi_profiler.profile_residuals``) and computes
    # the depth signal in Z-score space, with mu_t auto-anchored at
    # ``profile.mu_arch``.  In auto mode ``norm_base`` / ``mu_low`` /
    # ``mu_span`` are ignored at runtime; they are kept only as fallback
    # constants for the static-mode regression.
    # ----------------------------------------------------------------------
    profile_mode: Literal["static", "auto"] = "auto"

    # Z-score clamp range (D-S-5 in plan).  Avoids depth signal blow-up
    # on rare degenerate prompts where one layer norm is far in the tail.
    z_clamp: float = 3.0
    # Auto-mode mu drift coefficient: mu_t = mu_arch + c * (z_depth - 0.5) * L
    auto_mu_c: float = 0.2

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

    # ------------------------------------------------------------------
    # ECOR routing (W-T3 round 2, opt-in).  When ``use_ecor=False`` (default)
    # the production path is bit-for-bit identical to v3.4.  When
    # ``use_ecor=True``, ``apply_lopi`` routes the readout through
    # ``lopi_inject.lopi_inject`` with ``ecor_cfg`` (defaults to
    # ``ECORConfig()`` ⇒ additive early-return — also bit-equal).  The
    # purpose of the flag is to enable end-to-end NLL A/B between additive
    # and rotated injection without a code edit at the call site.
    # ------------------------------------------------------------------
    use_ecor: bool = False
    ecor_cfg: Any = None  # Optional[ECORConfig]; lazy-typed to avoid import cycle

    def asdict(self) -> dict:
        ecor_dict = None
        if self.ecor_cfg is not None:
            try:
                ecor_dict = {
                    "enabled": self.ecor_cfg.enabled,
                    "soft_blend": self.ecor_cfg.soft_blend,
                    "k": self.ecor_cfg.k,
                    "per_head": self.ecor_cfg.per_head,
                    "direction_eps": self.ecor_cfg.direction_eps,
                    "max_theta_frac": self.ecor_cfg.max_theta_frac,
                }
            except AttributeError:
                ecor_dict = repr(self.ecor_cfg)
        return {
            "enabled": self.enabled,
            "orthogonal": self.orthogonal,
            "gaussian": self.gaussian,
            "derivative": self.derivative,
            "profile_mode": self.profile_mode,
            "z_clamp": self.z_clamp,
            "auto_mu_c": self.auto_mu_c,
            "k_gate": self.k_gate,
            "theta_gate": self.theta_gate,
            "kappa_depth": self.kappa_depth,
            "beta_sigma": self.beta_sigma,
            "norm_base": self.norm_base,
            "mu_low": self.mu_low,
            "mu_span": self.mu_span,
            "sigma_floor": self.sigma_floor,
            "eps": self.eps,
            "use_ecor": self.use_ecor,
            "ecor_cfg": ecor_dict,
        }


# ---------------------------------------------------------------------------
# Per-call state — owned by the bank, mutated by the patched forward


@dataclass
class LOPIState:
    """Cross-step LOPI scratchpad held on the bank instance.

    All tensors are small (per-head Q vectors, per-layer scalar norms) so
    keeping them around for every decoded token is cheap.  The state is
    cleared by ``reset()`` when a new conversation / write-pass begins.

    Causality (Phase S, B1 fix)
    ---------------------------
    The v3.4 implementation read ``prev_residual_norms[layer_idx]`` for every
    layer of step ``t`` and *also* wrote the same dict at the end of each
    layer's forward.  Layer N+1 therefore saw the *current step*'s norm of
    layer N, not the t-1 snapshot the docstring promised -- the "causality
    trick" was broken for every layer above 0.  We now keep two dicts:

    * ``prev_residual_norms`` : the **frozen snapshot from step t-1** that
      every layer of step t reads.  Read-only inside a forward pass.
    * ``pending_residual_norms`` : the dict layers of the *current* step
      write into.  Promoted to ``prev_residual_norms`` by ``commit_step()``
      between successive forward calls.

    ``profile`` (Phase S)
    ---------------------
    Optional ``LOPIProfile`` populated by ``lopi_profiler.profile_residuals``.
    When present and ``cfg.profile_mode == "auto"`` the depth signal is
    computed in Z-score space and ``mu_t`` is auto-anchored at
    ``profile.mu_arch``.  The profile is also persisted by
    ``bank_persistence.save_bank``.
    """

    num_layers: int

    # Causality trick: caches from time-step t-1
    prev_q_per_layer: dict = field(default_factory=dict)         # layer_idx -> (B, H, T, D)
    prev_residual_norms: dict = field(default_factory=dict)      # layer_idx -> float (read-only this step)
    pending_residual_norms: dict = field(default_factory=dict)   # layer_idx -> float (written this step)

    # Phase S — auto-calibration profile (None => static mode)
    profile: Any = None  # LOPIProfile

    # Legacy mHC σ-max running mean (Phase R; B3: never wired by the bank
    # in production paths, kept only for ablation backward compat).
    mhc_sigma_max_running: float = 0.0
    mhc_sigma_count: int = 0

    def reset(self) -> None:
        self.prev_q_per_layer = {}
        self.prev_residual_norms = {}
        self.pending_residual_norms = {}
        self.mhc_sigma_max_running = 0.0
        self.mhc_sigma_count = 0
        # ``profile`` survives reset -- it is a per-architecture constant.

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

    def commit_step(self) -> None:
        """Promote pending norms (current step) to the frozen snapshot.

        Bank patcher MUST call this exactly once between two successive
        forward passes so layer 0 of step ``t+1`` reads layer 0 .. L-1 of
        step ``t``.  Without this swap the "causality trick" reduces to
        intra-step contamination (B1 in plan.md).
        """
        if self.pending_residual_norms:
            self.prev_residual_norms = dict(self.pending_residual_norms)
            self.pending_residual_norms = {}


# ---------------------------------------------------------------------------
# Pure-tensor primitives (test-friendly, no module state)


def derivative_gate(q_t: torch.Tensor, q_prev: Optional[torch.Tensor],
                    k: float, theta: float) -> torch.Tensor:
    """Compute gamma_t = sigmoid(k * (||Q_t - Q_{t-1}||_2 - theta)).

    Returns a tensor of shape (B, H, T, 1) so it broadcasts over D.  When
    ``q_prev is None`` (t = 0 or after reset) gamma_t collapses to 1 — this
    matches PREREGISTRATION §3.5 (legacy behavior at session boundaries).
    A shape mismatch between ``q_t`` and ``q_prev`` (e.g. a new prompt of
    different length) is treated as a session boundary as well.
    """
    if q_prev is None or q_prev.shape != q_t.shape:
        # Use a tensor (not python 1.0) so autograd / dtype propagate cleanly.
        return torch.ones((*q_t.shape[:-1], 1), dtype=q_t.dtype, device=q_t.device)
    delta_q = torch.linalg.vector_norm(q_t - q_prev, ord=2, dim=-1, keepdim=True)
    # In bf16/fp16 the sigmoid argument range matters; cast to fp32 for the
    # transcendental, then back. Cheap on every backend.
    arg = (delta_q.float() - theta) * k
    gamma = torch.sigmoid(arg).to(q_t.dtype)
    return gamma


def _z_depth_signal(state: LOPIState, cfg: LOPIConfig) -> float:
    """Compute the architecture-invariant depth signal d_t in (0, 1).

    ``d_t = sigmoid(k * mean_l Z_t(l))`` where ``Z_t(l) = (N_t(l) - mu_base(l))
    / (sigma_base(l) + eps)`` and ``N_t(l)`` is the t-1 snapshot of the
    layer-l residual L2 norm (Phase S, B1 causality fix).

    When the snapshot is empty (cold start, t=0) we return 0.5 — a neutral
    signal that anchors ``mu_t`` exactly at ``profile.mu_arch`` (B2 fix:
    v3.4 returned 0.12 here, biasing the first step toward shallow
    layers).
    """
    profile = state.profile
    if profile is None or not profile.mu_base or not state.prev_residual_norms:
        return 0.5
    L = profile.num_layers
    z_sum = 0.0
    z_count = 0
    for layer_idx, norm in state.prev_residual_norms.items():
        if not (0 <= layer_idx < L):
            continue
        mu = profile.mu_base[layer_idx]
        sigma = profile.sigma_base[layer_idx]
        denom = sigma + cfg.eps
        z = (float(norm) - mu) / denom
        # Clamp Z to (-z_clamp, +z_clamp) — D-S-5
        z = max(-cfg.z_clamp, min(cfg.z_clamp, z))
        z_sum += z
        z_count += 1
    if z_count == 0:
        return 0.5
    z_mean = z_sum / z_count
    arg = cfg.kappa_depth * z_mean
    # Pure float sigmoid — caller wraps in torch tensor.
    if arg >= 0:
        e = pow(2.718281828, -arg)
        return 1.0 / (1.0 + e)
    e = pow(2.718281828, arg)
    return e / (1.0 + e)


def layer_gaussian_weight(layer_idx: int, num_layers: int,
                          avg_prev_norm: float, mhc_sigma_max: float,
                          cfg: LOPIConfig,
                          device: torch.device, dtype: torch.dtype,
                          *, state: Optional[LOPIState] = None) -> torch.Tensor:
    """Compute w(ell, t) for the current layer.

    Returns a 0-dim tensor that broadcasts against (B, H, T, 1).

    Auto mode (Phase S, ``cfg.profile_mode == "auto"`` with a profile)
    -----------------------------------------------------------------
    ``mu_t = profile.mu_arch + auto_mu_c * (d_t - 0.5) * L``  where ``d_t``
    is the Z-score depth signal.  ``sigma_t = (L/6) * profile.eta_sigma``
    -- the dead ``mhc_sigma_max`` knob (B3 fix) is dropped from the
    auto path.

    Static mode (legacy v3.4)
    -------------------------
    Same formula as before.  Used by the regression suite to verify that
    auto/static differ only by depth-signal source.
    """
    L = float(num_layers)
    use_auto = (cfg.profile_mode == "auto"
                and state is not None
                and state.profile is not None)

    if use_auto:
        profile = state.profile
        d_t = _z_depth_signal(state, cfg)
        mu_t = float(profile.mu_arch) + cfg.auto_mu_c * (d_t - 0.5) * L
        sigma_t = max((L / 6.0) * float(profile.eta_sigma), cfg.sigma_floor)
    else:
        depth_arg = cfg.kappa_depth * (avg_prev_norm / max(cfg.norm_base, cfg.eps) - 1.0)
        # cheap fp scalar sigmoid (exp inputs bounded by clamping not needed
        # here -- depth_arg in static mode is already O(1))
        if depth_arg >= 0:
            depth_t = 1.0 / (1.0 + pow(2.718281828, -depth_arg))
        else:
            e = pow(2.718281828, depth_arg)
            depth_t = e / (1.0 + e)
        mu_t = L * (cfg.mu_low + cfg.mu_span * depth_t)
        sigma_t = max((L / 6.0) * pow(2.718281828, -cfg.beta_sigma * float(mhc_sigma_max)),
                      cfg.sigma_floor)

    # Gaussian weight as a torch scalar (B4 fix: use torch.exp not python pow,
    # so the value lands in the correct device/dtype without a host->device sync).
    mu_t_t = torch.tensor(mu_t, device=device, dtype=torch.float32)
    sigma_t_t = torch.tensor(sigma_t, device=device, dtype=torch.float32)
    layer_t = torch.tensor(float(layer_idx), device=device, dtype=torch.float32)
    w = torch.exp(-((layer_t - mu_t_t) ** 2) / (2.0 * sigma_t_t * sigma_t_t))
    return w.to(dtype)


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
    alpha: float = 1.0,
) -> torch.Tensor:
    """Apply Dynamic LOPI to one attention layer's bank readout.

    The function is **pure with respect to its config**: the only side
    effect on ``state`` is the t -> t+1 cache update at the end (so the
    caller does not have to remember to call a separate ``state.step()``).

    Returns the new ``out_bank`` tensor of identical shape to the input.
    Caller should add it to ``out_orig`` exactly as before.
    """
    if not cfg.enabled or float(alpha) == 0.0:
        return out_bank_native

    # 1. Orthogonal projection (op on the *readouts*, head-wise).
    if cfg.orthogonal:
        m_perp = orthogonal_novelty(out_bank_native, v_ctx_readout, cfg.eps)
    else:
        m_perp = out_bank_native

    # Phase X.1: m_perp energy ratio diagnostic (zero overhead when off).
    import deltamemory.diagnostics as _diag_mod  # noqa: PLC0415
    if _diag_mod._RECORDER is not None:
        _diag_mod._RECORDER.record_m_perp_ratio(layer_idx, m_perp, out_bank_native)

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
            state=state,
        )
    else:
        w_ell = torch.tensor(1.0, device=m_perp.device, dtype=m_perp.dtype)

    # 3. Derivative gate (per-(B,H,T) tensor, broadcasts over D).
    if cfg.derivative:
        q_prev = state.prev_q_per_layer.get(layer_idx)
        gamma_t = derivative_gate(q_post, q_prev, cfg.k_gate, cfg.theta_gate)
    else:
        gamma_t = torch.tensor(1.0, device=m_perp.device, dtype=m_perp.dtype)

    # Phase X.1: LOPI gate diagnostics (zero overhead when recorder is off).
    if _diag_mod._RECORDER is not None:
        _diag_mod._RECORDER.record_lopi_gamma_w(layer_idx, gamma_t, w_ell)

    out_bank_lopi = gamma_t * w_ell * m_perp

    if cfg.use_ecor and cfg.ecor_cfg is not None \
            and getattr(cfg.ecor_cfg, "enabled", False) \
            and getattr(cfg.ecor_cfg, "soft_blend", 0.0) != 0.0:
        # W-T3 round 2: route through the ECOR operator only when it would
        # actually do work.  Otherwise fall through to the bit-equal legacy
        # ``gamma_t · w_ell · m_perp`` value computed above (this preserves
        # strict bit-equality with v3.4 in the additive degenerate case).
        # The operator returns the *full* readout V_ctx + s·M_perp (or its
        # rotated blend); we subtract V_ctx so the caller's
        #     attn_out = out_orig + out_bank
        # is arithmetically equivalent.
        from deltamemory.memory.lopi_inject import lopi_inject as _lopi_inject

        v_full = _lopi_inject(
            V_ctx=v_ctx_readout,
            M_perp=m_perp,
            gamma_t=gamma_t,
            w_ell=w_ell,
            alpha_base=1.0,  # alpha already folded into m_perp via out_bank_native
            cfg=cfg.ecor_cfg,
        )
        out_bank_lopi = v_full - v_ctx_readout


    try:
        from deltamemory.security.audit import audit_event

        out_norm = float(torch.linalg.vector_norm(out_bank_lopi.detach().float()).item())
        ctx_norm = float(torch.linalg.vector_norm(v_ctx_readout.detach().float()).item())
        gate_mean = float(gamma_t.detach().float().mean().item())
        audit_event(
            event_type="inject",
            injector="lopi",
            layer=layer_idx,
            alpha=float(alpha),
            signal_summary={
                "steer_norm": out_norm,
                "drift_ratio": out_norm / (ctx_norm + 1e-10),
                "gate_mean": gate_mean,
            },
            vector_tensor=gamma_t,
        )
    except Exception:
        pass

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
