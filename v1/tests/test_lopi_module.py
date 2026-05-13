"""Unit tests for Dynamic LOPI v3.3 (Phase R, R-2 smoke gate).

Coverage
--------
1. ``LOPIConfig`` defaults (enabled=False) ⇒ ``apply_lopi`` is identity.
2. ``LOPIConfig(enabled=True, all-components-off)`` ⇒ ``apply_lopi`` is identity.
3. Orthogonal projection: M_perp ⟂ V_ctx within numerical tolerance.
4. Orthogonal projection edge case: V_ctx == 0 ⇒ M_perp == M_V.
5. Derivative gate: q_prev=None ⇒ gamma_t == 1 everywhere.
6. Derivative gate: q_t == q_prev ⇒ gamma_t == sigmoid(-k*theta) (small).
7. Layer Gaussian: w(mu, t) == 1 (peak at center).
8. ``LOPIState.update_mhc_sigma`` running mean correctness.
9. End-to-end shape preservation through ``apply_lopi``.
"""

from __future__ import annotations

import math

import pytest
import torch

from deltamemory.memory.lopi import (
    LOPIConfig,
    LOPIState,
    apply_lopi,
    derivative_gate,
    layer_gaussian_weight,
    orthogonal_novelty,
)


def _rand(*shape, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype)


# ---------------------------------------------------------------------------
# Bit-equal degeneracy gates


def test_lopi_disabled_is_identity():
    cfg = LOPIConfig()  # enabled=False
    state = LOPIState(num_layers=4)
    out_bank = _rand(2, 3, 5, 8, seed=1)
    v_ctx = _rand(2, 3, 5, 8, seed=2)
    q_post = _rand(2, 3, 5, 8, seed=3)
    result = apply_lopi(out_bank, v_ctx, q_post, layer_idx=2, state=state, cfg=cfg)
    assert torch.equal(result, out_bank), "LOPI off must be byte-identical"


def test_lopi_all_components_off_is_identity():
    cfg = LOPIConfig(enabled=True, orthogonal=False, gaussian=False, derivative=False)
    state = LOPIState(num_layers=4)
    out_bank = _rand(2, 3, 5, 8, seed=4)
    v_ctx = _rand(2, 3, 5, 8, seed=5)
    q_post = _rand(2, 3, 5, 8, seed=6)
    result = apply_lopi(out_bank, v_ctx, q_post, layer_idx=2, state=state, cfg=cfg)
    # gamma=1, w=1, M_perp=M_V → identity within fp tolerance.
    torch.testing.assert_close(result, out_bank, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Orthogonal projection


def test_orthogonal_projection_is_perpendicular():
    m_v = _rand(2, 3, 5, 8, seed=7)
    v_ctx = _rand(2, 3, 5, 8, seed=8)
    m_perp = orthogonal_novelty(m_v, v_ctx, eps=1e-8)
    # Inner product along D should be ~0.
    inner = (m_perp * v_ctx).sum(dim=-1)
    torch.testing.assert_close(inner, torch.zeros_like(inner), atol=1e-5, rtol=0)


def test_orthogonal_projection_zero_v_ctx_returns_m_v():
    m_v = _rand(1, 1, 1, 4, seed=9)
    v_ctx = torch.zeros_like(m_v)
    m_perp = orthogonal_novelty(m_v, v_ctx, eps=1e-6)
    torch.testing.assert_close(m_perp, m_v)


# ---------------------------------------------------------------------------
# Derivative gate


def test_derivative_gate_no_prev_returns_one():
    q_t = _rand(2, 3, 5, 8, seed=10)
    gamma = derivative_gate(q_t, q_prev=None, k=5.0, theta=0.5)
    assert gamma.shape == (2, 3, 5, 1)
    torch.testing.assert_close(gamma, torch.ones_like(gamma))


def test_derivative_gate_zero_delta_is_small():
    q_t = _rand(1, 1, 1, 4, seed=11)
    q_prev = q_t.clone()
    gamma = derivative_gate(q_t, q_prev=q_prev, k=5.0, theta=0.5)
    expected = 1.0 / (1.0 + math.exp(5.0 * 0.5))  # sigmoid(-2.5)
    assert abs(float(gamma.item()) - expected) < 1e-5


# ---------------------------------------------------------------------------
# Layer Gaussian


def test_layer_gaussian_peaks_at_center():
    cfg = LOPIConfig(kappa_depth=2.0, beta_sigma=2.0, norm_base=10.0,
                     mu_low=0.3, mu_span=0.5)
    L = 24
    avg_norm = 10.0  # depth_t = sigmoid(0) = 0.5 → mu_t = L * 0.55 = 13.2
    mhc_sigma = 0.0
    expected_mu = L * (cfg.mu_low + cfg.mu_span * 0.5)
    w_peak = layer_gaussian_weight(int(round(expected_mu)), L, avg_norm, mhc_sigma,
                                   cfg, torch.device("cpu"), torch.float32)
    w_far = layer_gaussian_weight(0, L, avg_norm, mhc_sigma,
                                  cfg, torch.device("cpu"), torch.float32)
    assert float(w_peak) > 0.95
    assert float(w_far) < float(w_peak)


# ---------------------------------------------------------------------------
# State management


def test_lopi_state_running_sigma_mean():
    state = LOPIState(num_layers=4)
    state.update_mhc_sigma(1.0)
    state.update_mhc_sigma(3.0)
    state.update_mhc_sigma(5.0)
    assert abs(state.mhc_sigma_max_running - 3.0) < 1e-6
    assert state.mhc_sigma_count == 3


def test_lopi_state_reset_clears():
    state = LOPIState(num_layers=4)
    state.update_mhc_sigma(2.0)
    state.prev_residual_norms[0] = 1.5
    state.prev_q_per_layer[0] = torch.zeros(1)
    state.reset()
    assert state.mhc_sigma_count == 0
    assert state.mhc_sigma_max_running == 0.0
    assert state.prev_residual_norms == {}
    assert state.prev_q_per_layer == {}


def test_apply_lopi_caches_q_for_next_step():
    cfg = LOPIConfig(enabled=True, orthogonal=True, gaussian=False, derivative=True)
    state = LOPIState(num_layers=4)
    q1 = _rand(1, 1, 1, 4, seed=20)
    out_bank = _rand(1, 1, 1, 4, seed=21)
    v_ctx = _rand(1, 1, 1, 4, seed=22)
    apply_lopi(out_bank, v_ctx, q1, layer_idx=2, state=state, cfg=cfg)
    assert 2 in state.prev_q_per_layer
    torch.testing.assert_close(state.prev_q_per_layer[2], q1)


# ---------------------------------------------------------------------------
# End-to-end shape


def test_apply_lopi_preserves_shape():
    cfg = LOPIConfig(enabled=True)
    state = LOPIState(num_layers=12)
    state.prev_residual_norms = {i: 8.0 for i in range(12)}
    out_bank = _rand(2, 4, 6, 16, seed=30)
    v_ctx = _rand(2, 4, 6, 16, seed=31)
    q_post = _rand(2, 4, 6, 16, seed=32)
    result = apply_lopi(out_bank, v_ctx, q_post, layer_idx=6, state=state, cfg=cfg)
    assert result.shape == out_bank.shape
    assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# R-2 ablation grid (A0..A4) parametric smoke


@pytest.mark.parametrize("variant,cfg_kwargs", [
    ("A0", dict(enabled=False)),
    ("A1", dict(enabled=True, orthogonal=True, gaussian=False, derivative=False)),
    ("A2", dict(enabled=True, orthogonal=True, gaussian=True, derivative=False)),
    ("A3", dict(enabled=True, orthogonal=True, gaussian=True, derivative=True)),
    ("A4", dict(enabled=True, orthogonal=False, gaussian=True, derivative=True)),
])
def test_lopi_ablation_variants_finite_and_distinct(variant, cfg_kwargs):
    """R-2 smoke: every ablation variant produces finite output, and the
    non-trivial variants (A1..A4) do *not* equal A0 unless their config
    happens to reduce to identity (which is exercised in earlier tests).
    """
    cfg = LOPIConfig(**cfg_kwargs)
    state = LOPIState(num_layers=12)
    state.prev_residual_norms = {i: 8.0 for i in range(12)}
    out_bank = _rand(1, 2, 4, 8, seed=40)
    v_ctx = _rand(1, 2, 4, 8, seed=41)
    q_post = _rand(1, 2, 4, 8, seed=42)
    out = apply_lopi(out_bank, v_ctx, q_post, layer_idx=6, state=state, cfg=cfg)
    assert torch.isfinite(out).all(), f"{variant}: non-finite output"
    if variant == "A0":
        assert torch.equal(out, out_bank)
    else:
        # Each enabled component must perturb the output.
        diff = (out - out_bank).abs().max().item()
        assert diff > 0.0, f"{variant}: output identical to A0 baseline"


def test_lopi_two_step_derivative_drops_gamma():
    """When Q evolves significantly between steps, gamma_t opens (>0.5).
    When Q is stable, gamma_t closes (<0.5). Tests cross-step state plumbing."""
    cfg = LOPIConfig(enabled=True, orthogonal=False, gaussian=False, derivative=True,
                     k_gate=5.0, theta_gate=0.5)
    state = LOPIState(num_layers=4)
    out_bank = torch.ones(1, 1, 1, 4)
    v_ctx = torch.ones(1, 1, 1, 4) * 2.0

    # Step 0: no prev → gamma=1
    q0 = torch.zeros(1, 1, 1, 4)
    out0 = apply_lopi(out_bank, v_ctx, q0, layer_idx=2, state=state, cfg=cfg)
    assert torch.allclose(out0, out_bank)

    # Step 1: large jump → gamma → 1
    q1 = torch.ones(1, 1, 1, 4) * 5.0  # ‖Δ‖ ≈ 10 >> theta
    out1 = apply_lopi(out_bank, v_ctx, q1, layer_idx=2, state=state, cfg=cfg)
    assert float(out1.mean()) > 0.95, f"large jump should open gate, got {out1.mean()}"

    # Step 2: tiny jump → gamma small
    q2 = q1 + 0.01
    out2 = apply_lopi(out_bank, v_ctx, q2, layer_idx=2, state=state, cfg=cfg)
    assert float(out2.mean()) < 0.5, f"stable Q should close gate, got {out2.mean()}"


# ---------------------------------------------------------------------------
# Bank integration: lopi_cfg defaults to disabled, bank stays bit-equal


def test_bank_default_lopi_disabled():
    from deltamemory.memory.attn_native_bank import AttnNativeBank

    bank = AttnNativeBank(num_layers=4, num_kv_heads=2, head_dim=8)
    assert bank.lopi_cfg is not None
    assert bank.lopi_cfg.enabled is False
    assert bank.lopi_state.num_layers == 4
