"""Phase S — U-LOPI auto-calibration integration tests.

Coverage
--------
1. Static-mode ≡ legacy v3.4: bit-equal to running with profile_mode='static'
   while no profile attached.
2. Auto-mode degrades to static when ``state.profile is None`` (back-compat).
3. Cold-start (profile attached but pending_residual_norms empty) ⇒
   ``d_t == 0.5`` and ``mu_t == profile.mu_arch`` exactly.
4. ``commit_step`` promotes pending → prev and clears pending (B1 fix).
5. Z-score clamp: extreme prev norms saturate at z_clamp instead of blowing up.
6. Auto-mode at the architecture's own ``mu_arch`` layer with ``d_t=0.5``
   produces ``w == 1`` (peak of the Gaussian).
"""
from __future__ import annotations

import math

import pytest
import torch

from deltamemory.memory.lopi import (
    LOPIConfig,
    LOPIState,
    _z_depth_signal,
    layer_gaussian_weight,
)
from deltamemory.memory.lopi_profiler import LOPIProfile


def _make_profile(num_layers=12, mu_arch=6, eta=1.0, mu_base=None,
                  sigma_base=None) -> LOPIProfile:
    if mu_base is None:
        mu_base = [10.0] * num_layers
    if sigma_base is None:
        sigma_base = [1.0] * num_layers
    return LOPIProfile(
        model_name="_test/fake",
        num_layers=num_layers,
        mu_base=list(mu_base),
        sigma_base=list(sigma_base),
        mu_arch=mu_arch,
        eta_sigma=eta,
        profile_corpus_sha="deadbeef" * 2,
        n_prompts=10,
        dtype="float32",
    )


def test_auto_without_profile_falls_back_to_static():
    """Default cfg has profile_mode='auto' but no profile ⇒ static branch."""
    cfg_auto = LOPIConfig(profile_mode="auto")
    cfg_static = LOPIConfig(profile_mode="static")
    state = LOPIState(num_layers=8)  # profile=None
    w_auto = layer_gaussian_weight(
        layer_idx=4, num_layers=8, avg_prev_norm=0.0, mhc_sigma_max=0.0,
        cfg=cfg_auto, device=torch.device("cpu"), dtype=torch.float32,
        state=state,
    )
    w_static = layer_gaussian_weight(
        layer_idx=4, num_layers=8, avg_prev_norm=0.0, mhc_sigma_max=0.0,
        cfg=cfg_static, device=torch.device("cpu"), dtype=torch.float32,
        state=None,
    )
    torch.testing.assert_close(w_auto, w_static, rtol=0, atol=0)


def test_cold_start_anchors_at_mu_arch():
    """Profile attached but no t-1 snapshot ⇒ d_t=0.5, mu_t==mu_arch exactly."""
    profile = _make_profile(num_layers=12, mu_arch=7)
    state = LOPIState(num_layers=12, profile=profile)
    cfg = LOPIConfig(profile_mode="auto")
    d = _z_depth_signal(state, cfg)
    assert d == 0.5
    # At layer == mu_arch with d=0.5 -> mu_t == mu_arch -> w == 1.0
    w = layer_gaussian_weight(
        layer_idx=7, num_layers=12, avg_prev_norm=0.0, mhc_sigma_max=0.0,
        cfg=cfg, device=torch.device("cpu"), dtype=torch.float32, state=state,
    )
    torch.testing.assert_close(w, torch.tensor(1.0), rtol=1e-5, atol=1e-5)


def test_commit_step_promotes_pending():
    state = LOPIState(num_layers=4)
    state.pending_residual_norms[0] = 1.5
    state.pending_residual_norms[1] = 2.5
    state.commit_step()
    assert state.prev_residual_norms == {0: 1.5, 1: 2.5}
    assert state.pending_residual_norms == {}


def test_commit_step_noop_on_empty():
    state = LOPIState(num_layers=4)
    state.prev_residual_norms[0] = 9.9   # should NOT be wiped
    state.commit_step()
    assert state.prev_residual_norms == {0: 9.9}


def test_z_depth_clamps_extreme_norms():
    """Huge prev norms must not blow d_t past sigmoid saturation."""
    profile = _make_profile(num_layers=4, mu_arch=2,
                            mu_base=[1.0, 1.0, 1.0, 1.0],
                            sigma_base=[0.01, 0.01, 0.01, 0.01])
    state = LOPIState(num_layers=4, profile=profile)
    state.prev_residual_norms = {0: 1e6, 1: 1e6, 2: 1e6, 3: 1e6}
    d = _z_depth_signal(state, LOPIConfig(profile_mode="auto", z_clamp=3.0))
    # With z clamped to +3 and kappa=2.0, d == sigmoid(6) ≈ 0.9975, not 1.0
    expected = 1.0 / (1.0 + math.exp(-6.0))
    assert abs(d - expected) < 1e-4


def test_auto_mu_t_matches_formula_at_d_neq_half():
    """When prev norms exceed mu_base, d_t > 0.5 ⇒ mu_t > mu_arch."""
    profile = _make_profile(
        num_layers=10, mu_arch=4,
        mu_base=[5.0] * 10,
        sigma_base=[1.0] * 10,
    )
    state = LOPIState(num_layers=10, profile=profile)
    # All prev norms 5 + 2*sigma ⇒ Z = 2 (under z_clamp=3)
    state.prev_residual_norms = {i: 7.0 for i in range(10)}
    cfg = LOPIConfig(profile_mode="auto", z_clamp=3.0,
                     auto_mu_c=0.2, kappa_depth=2.0)
    d = _z_depth_signal(state, cfg)
    expected_d = 1.0 / (1.0 + math.exp(-2.0 * 2.0))  # sigmoid(4)
    assert abs(d - expected_d) < 1e-4
    # Peak of Gaussian: at ell == mu_arch + 0.2*(d-0.5)*L
    L = 10.0
    peak_layer = profile.mu_arch + cfg.auto_mu_c * (d - 0.5) * L
    # w at the closest integer layer should be < 1 (peak is between layers).
    nearest = int(round(peak_layer))
    w_peak = layer_gaussian_weight(
        layer_idx=nearest, num_layers=10, avg_prev_norm=0.0, mhc_sigma_max=0.0,
        cfg=cfg, device=torch.device("cpu"), dtype=torch.float32, state=state,
    )
    # Far from peak -> much smaller.
    w_far = layer_gaussian_weight(
        layer_idx=0, num_layers=10, avg_prev_norm=0.0, mhc_sigma_max=0.0,
        cfg=cfg, device=torch.device("cpu"), dtype=torch.float32, state=state,
    )
    assert w_peak.item() > w_far.item()


def test_static_mode_preserves_legacy_formula():
    """profile_mode='static' must produce the v3.4 numerical answer."""
    cfg = LOPIConfig(profile_mode="static",
                     norm_base=10.0, kappa_depth=2.0, beta_sigma=2.0,
                     mu_low=0.3, mu_span=0.5)
    L = 12
    avg_prev = 12.0  # depth_arg = 2*(1.2-1)=0.4 -> sigmoid(0.4) = 0.5987
    w = layer_gaussian_weight(
        layer_idx=6, num_layers=L, avg_prev_norm=avg_prev,
        mhc_sigma_max=0.0,
        cfg=cfg, device=torch.device("cpu"), dtype=torch.float32, state=None,
    )
    depth_t = 1.0 / (1.0 + math.exp(-0.4))
    mu_t = L * (0.3 + 0.5 * depth_t)
    sigma_t = (L / 6.0) * math.exp(0.0)  # sigma_max=0 ⇒ exp(0)=1
    expected = math.exp(-((6 - mu_t) ** 2) / (2 * sigma_t * sigma_t))
    assert abs(w.item() - expected) < 1e-5
