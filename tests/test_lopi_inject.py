"""Phase X.7 — Tests for lopi_inject (ECOR opt-in injection operator).

Coverage (8 cases)
------------------
1. test_disabled_is_bit_equal_to_additive
       ECORConfig(enabled=False) produces max-abs-diff = 0 vs explicit V_ctx + s·M_perp.
2. test_soft_blend_zero_is_bit_equal_to_additive
       soft_blend=0 with enabled=True also produces max-abs-diff = 0.
3. test_soft_blend_one_at_low_s_is_close_to_V_ctx
       At s≈0 (gamma=0), V_rot ≈ V_ctx (cos(0)=1, sin(0)=0).
4. test_direction_fallback_fires
       When ‖M_perp‖/‖V_ctx‖ < direction_eps, output equals V_add exactly.
5. test_max_theta_caps_at_60_deg
       At very large s, theta → max_theta_frac·π = π/3; cos(theta) > 0.49.
6. test_per_head_shapes
       V_ctx of shape (2, 8, 16, 64) (B=2, H=8, T=16, D_head=64), output same shape.
7. test_norm_preservation_property
       At soft_blend=1 with truly orthogonal M_perp, ‖V_out‖ ≈ ‖V_ctx‖ within 1e-5.
8. test_alpha_zero_and_zero_M_perp_return_V_ctx
       M_perp=0 (no bank) OR s=0 (gate off) returns V_ctx unchanged.
"""

from __future__ import annotations

import math

import pytest
import torch

from deltamemory.memory.lopi_inject import ECORConfig, lopi_inject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand(*shape, seed: int = 42, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype)


def _make_orthogonal(M_rand: torch.Tensor, V_ctx: torch.Tensor) -> torch.Tensor:
    """Return the component of M_rand orthogonal to V_ctx (last dim projection)."""
    v_norm_sq = (V_ctx * V_ctx).sum(dim=-1, keepdim=True) + 1e-8
    dot = (M_rand * V_ctx).sum(dim=-1, keepdim=True)
    return M_rand - (dot / v_norm_sq) * V_ctx


def _scalar_w() -> torch.Tensor:
    return torch.tensor(1.0)


def _scalar_gamma() -> torch.Tensor:
    return torch.tensor(1.0)


# ---------------------------------------------------------------------------
# 1. ECORConfig(enabled=False) → bit-equal additive
# ---------------------------------------------------------------------------


def test_disabled_is_bit_equal_to_additive():
    """disabled path must be EXACTLY V_ctx + s·M_perp (no float drift)."""
    B, T, D = 3, 7, 32
    V_ctx = _rand(B, T, D, seed=1)
    M_perp = _rand(B, T, D, seed=2)
    gamma_t = torch.tensor(0.7)
    w_ell = torch.tensor(1.5)
    alpha = 0.8

    s = gamma_t * w_ell * alpha
    expected = V_ctx + s * M_perp

    cfg = ECORConfig(enabled=False)
    result = lopi_inject(V_ctx, M_perp, gamma_t, w_ell, alpha_base=alpha, cfg=cfg)

    assert (result - expected).abs().max().item() == 0.0, (
        "ECORConfig(enabled=False) must be bit-equal to additive baseline"
    )


# ---------------------------------------------------------------------------
# 2. soft_blend=0 → bit-equal additive (even when enabled=True)
# ---------------------------------------------------------------------------


def test_soft_blend_zero_is_bit_equal_to_additive():
    """soft_blend=0 with enabled=True must be bit-equal to additive baseline."""
    B, T, D = 4, 5, 16
    V_ctx = _rand(B, T, D, seed=10)
    M_perp = _rand(B, T, D, seed=11)
    gamma_t = torch.tensor(0.9)
    w_ell = torch.tensor(2.0)
    alpha = 1.0

    s = gamma_t * w_ell * alpha
    expected = V_ctx + s * M_perp

    cfg = ECORConfig(enabled=True, soft_blend=0.0)
    result = lopi_inject(V_ctx, M_perp, gamma_t, w_ell, alpha_base=alpha, cfg=cfg)

    assert (result - expected).abs().max().item() == 0.0, (
        "soft_blend=0 must be bit-equal to additive baseline regardless of enabled flag"
    )


# ---------------------------------------------------------------------------
# 3. At s≈0 (gamma=0), V_rot ≈ V_ctx
# ---------------------------------------------------------------------------


def test_soft_blend_one_at_low_s_is_close_to_V_ctx():
    """At gamma=0 (s=0), theta=0, cos(0)=1, sin(0)=0 → V_rot ≈ V_ctx."""
    B, T, D = 2, 6, 24
    V_ctx = _rand(B, T, D, seed=20)
    M_perp = _rand(B, T, D, seed=21)
    gamma_t = torch.tensor(0.0)    # gate fully closed → s = 0
    w_ell = torch.tensor(1.0)

    cfg = ECORConfig(enabled=True, soft_blend=1.0)
    result = lopi_inject(V_ctx, M_perp, gamma_t, w_ell, alpha_base=1.0, cfg=cfg)

    max_diff = (result - V_ctx).abs().max().item()
    assert max_diff < 1e-6, f"At s=0 (gamma=0), V_out should ≈ V_ctx; got max_diff={max_diff}"


# ---------------------------------------------------------------------------
# 4. Direction fallback fires when ‖M_perp‖/‖V_ctx‖ < direction_eps
# ---------------------------------------------------------------------------


def test_direction_fallback_fires():
    """When M_perp ≈ 0 (‖M_perp‖/‖V_ctx‖ < direction_eps), output = V_add."""
    B, T, D = 2, 4, 16
    V_ctx = _rand(B, T, D, seed=30)
    # Make M_perp tiny so norm_m / norm_v << direction_eps
    M_perp = V_ctx * 1e-7   # same direction, tiny magnitude

    gamma_t = torch.tensor(1.0)
    w_ell = torch.tensor(1.0)
    alpha = 1.0
    s = gamma_t * w_ell * alpha
    expected_V_add = V_ctx + s * M_perp

    cfg = ECORConfig(enabled=True, soft_blend=1.0, direction_eps=1e-3)
    result = lopi_inject(V_ctx, M_perp, gamma_t, w_ell, alpha_base=alpha, cfg=cfg)

    # All tokens should have fallen back to V_add
    max_diff = (result - expected_V_add).abs().max().item()
    assert max_diff == 0.0, (
        f"Direction fallback should produce exactly V_add; got max_diff={max_diff}"
    )


# ---------------------------------------------------------------------------
# 5. max_theta cap at π/3 (60°) prevents V_ctx erasure
# ---------------------------------------------------------------------------


def test_max_theta_caps_at_60_deg():
    """With large s, theta approaches max_theta_frac·π = π/3, NOT π/2.

    cos(π/3) = 0.5 > 0.49, so V_ctx is never fully erased.
    """
    D = 64
    V_ctx = _rand(1, 1, D, seed=40)
    M_perp_raw = _rand(1, 1, D, seed=41)
    M_perp = _make_orthogonal(M_perp_raw, V_ctx)

    # Very large gamma → s is huge → tanh saturates → theta ≈ max_theta_frac·π
    gamma_t = torch.tensor(1e6)
    w_ell = torch.tensor(1.0)

    cfg = ECORConfig(enabled=True, soft_blend=1.0, max_theta_frac=1.0 / 3.0, k=1.0)
    result = lopi_inject(V_ctx, M_perp, gamma_t, w_ell, alpha_base=1.0, cfg=cfg)

    # V_out = cos(theta)·V_ctx + sin(theta)·M_unit_scaled
    # cos(π/3) = 0.5 exactly; V_ctx coefficient must be > 0.49
    # We can estimate it from ‖V_out · V_ctx_unit‖ / ‖V_ctx‖
    V_ctx_unit = V_ctx / V_ctx.norm(dim=-1, keepdim=True)
    cos_component = (result * V_ctx_unit).sum(dim=-1) / V_ctx.norm(dim=-1)
    cos_val = cos_component.abs().min().item()

    assert cos_val > 0.49, (
        f"max_theta=π/3 → cos(theta)≥0.5>0.49 but got cos_component={cos_val:.4f}. "
        "V_ctx erasure at θ=π/2 should be prevented."
    )


# ---------------------------------------------------------------------------
# 6. Per-head shape: V_ctx (2, 8, 16, 64) → output same shape
# ---------------------------------------------------------------------------


def test_per_head_shapes():
    """V_ctx (B=2, H=8, T=16, D_head=64) should produce output of same shape."""
    B, H, T, D_head = 2, 8, 16, 64
    V_ctx = _rand(B, H, T, D_head, seed=50)
    M_perp = _rand(B, H, T, D_head, seed=51)
    gamma_t = _rand(B, H, T, 1, seed=52).abs()   # (B, H, T, 1) — LOPI-style gate shape
    w_ell = torch.tensor(0.5)

    for enabled, blend in [(False, 0.0), (True, 0.5), (True, 1.0)]:
        cfg = ECORConfig(enabled=enabled, soft_blend=blend)
        result = lopi_inject(V_ctx, M_perp, gamma_t, w_ell, alpha_base=1.0, cfg=cfg)
        assert result.shape == V_ctx.shape, (
            f"Output shape mismatch: got {result.shape}, expected {V_ctx.shape}"
        )


# ---------------------------------------------------------------------------
# 7. Norm preservation at soft_blend=1 with orthogonal M_perp
# ---------------------------------------------------------------------------


def test_norm_preservation_property():
    """‖V_out‖ ≈ ‖V_ctx‖ (within 1e-5) when M_perp ⊥ V_ctx and soft_blend=1.

    The ECOR formula:
        V_rot = cos(θ)·V_ctx + sin(θ)·(M_perp·‖V_ctx‖/‖M_perp‖)
    preserves ‖V_rot‖ = ‖V_ctx‖ iff M_perp ⊥ V_ctx and ‖M_perp‖ > 0.
    """
    B, T, D = 4, 8, 32
    V_ctx = _rand(B, T, D, seed=60)
    M_rand = _rand(B, T, D, seed=61)
    M_perp = _make_orthogonal(M_rand, V_ctx)

    gamma_t = torch.tensor(0.5)
    w_ell = torch.tensor(1.0)

    cfg = ECORConfig(enabled=True, soft_blend=1.0, k=1.0, direction_eps=0.0)
    result = lopi_inject(V_ctx, M_perp, gamma_t, w_ell, alpha_base=1.0, cfg=cfg)

    norm_in = V_ctx.norm(dim=-1)
    norm_out = result.norm(dim=-1)
    rel_err = ((norm_out - norm_in).abs() / norm_in.clamp_min(1e-8)).max().item()

    assert rel_err < 1e-5, (
        f"ECOR norm not preserved: relative error = {rel_err:.2e} > 1e-5. "
        "cos²+sin²=1 should guarantee norm preservation when M_perp ⊥ V_ctx."
    )


# ---------------------------------------------------------------------------
# 8. Zero M_perp or zero s → return V_ctx unchanged
# ---------------------------------------------------------------------------


def test_alpha_zero_and_zero_M_perp_return_V_ctx():
    """When M_perp=0 (no bank) or s=0 (gate off), output equals V_ctx."""
    B, T, D = 3, 5, 20
    V_ctx = _rand(B, T, D, seed=70)

    # Case A: M_perp = 0
    M_zero = torch.zeros_like(V_ctx)
    gamma_t = torch.tensor(1.0)
    w_ell = torch.tensor(1.0)

    cfg_ecor = ECORConfig(enabled=True, soft_blend=1.0)
    result_A = lopi_inject(V_ctx, M_zero, gamma_t, w_ell, alpha_base=1.0, cfg=cfg_ecor)
    assert (result_A - V_ctx).abs().max().item() == 0.0, (
        "With M_perp=0, output should equal V_ctx (direction fallback + zero addend)"
    )

    # Case B: s = 0 (gamma = 0)
    M_nonzero = _rand(B, T, D, seed=71)
    gamma_zero = torch.tensor(0.0)

    result_B = lopi_inject(V_ctx, M_nonzero, gamma_zero, w_ell, alpha_base=1.0, cfg=cfg_ecor)
    assert (result_B - V_ctx).abs().max().item() < 1e-6, (
        "With gamma=0 (s=0), theta=0, V_rot=V_ctx so output should ≈ V_ctx"
    )
