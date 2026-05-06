"""Tests for deltamemory.injection.safe_alpha (v0.7 empirical version).

The v0.6 cliff narrative was debunked; see
``runs/X7NL_full_v1_gemma4_31B/ALPHA_CLIFF_DEBUNK.md``. These tests
exercise the new empirical-sweep API plus the deprecated legacy shims.
"""
from __future__ import annotations

import math

import pytest

from deltamemory.injection.safe_alpha import (
    AlphaProbeResult,
    CLIFF_HI_DEFAULT,
    CLIFF_LO_DEFAULT,
    CLIFF_THRESHOLD_DEFAULT,
    DEPLOY_ALPHA_FLOOR_DEFAULT,
    NOISE_STD_BUDGET_DEFAULT,
    POST_CLIFF_ALPHA_DEFAULT,
    RECOMMENDED_ALPHA_DEFAULT,
    SafeAlphaScheduler,
    compute_safe_alpha_threshold,
    empirical_alpha_sweep,
    recommend_alpha_from_sweep,
    validate_scheduler_vs_naive,
)


def make_x7nl_like_probe(
    *,
    noisy_lo: float = 0.05,
    noisy_hi: float = 0.50,
    noisy_std: float = 1.7,
    plateau_mean: float = 0.365,
    plateau_std: float = 0.31,
):
    """Mimic the real X.7-NL behaviour: high-variance bowl + low-variance plateau."""
    def probe(alpha: float, seed: int) -> float:
        offset = ((seed * 37) % 7 - 3) / 3.0
        if alpha == 0.0:
            return 0.0
        if noisy_lo <= alpha <= noisy_hi:
            return plateau_mean - 1.0 + noisy_std * offset
        return plateau_mean + plateau_std * offset
    return probe


class TestSafeAlphaScheduler:
    def test_zero_is_safe(self):
        sched = SafeAlphaScheduler()
        assert sched.is_safe(0.0)
        assert sched.safe_alpha(0.0) == 0.0

    def test_above_floor_passes_through(self):
        sched = SafeAlphaScheduler(deploy_floor=0.7, recommended=1.0)
        assert sched.safe_alpha(0.7) == 0.7
        assert sched.safe_alpha(1.0) == 1.0
        assert sched.safe_alpha(2.0) == 2.0

    def test_below_floor_bumps_up(self):
        sched = SafeAlphaScheduler(deploy_floor=0.7, recommended=1.0)
        assert sched.safe_alpha(0.25) == 1.0
        assert sched.safe_alpha(0.5) == 1.0

    def test_schedule_endpoints(self):
        sched = SafeAlphaScheduler()
        out = sched.schedule(5, 0.0, 1.0)
        assert len(out) == 5
        assert out[0] == 0.0
        assert out[-1] == 1.0

    def test_disabled_floor_is_noop(self):
        sched = SafeAlphaScheduler(deploy_floor=0.0)
        for a in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
            assert sched.safe_alpha(a) == a


class TestEmpiricalSweep:
    def test_requires_two_seeds(self):
        with pytest.raises(ValueError, match=r"≥2 seeds"):
            empirical_alpha_sweep(lambda a, s: 0.0, seeds=[0])

    def test_detects_noisy_region(self):
        probe = make_x7nl_like_probe()
        sweep = empirical_alpha_sweep(
            probe,
            alphas=[0.25, 0.5, 1.0, 1.5],
            seeds=[0, 1, 2, 3, 4],
            noise_std_budget=1.0,
        )
        assert {r.alpha for r in sweep} == {0.25, 0.5, 1.0, 1.5}
        noisy = [r for r in sweep if r.alpha in (0.25, 0.5)]
        plateau = [r for r in sweep if r.alpha in (1.0, 1.5)]
        assert all(r.is_noisy for r in noisy)
        assert all(not r.is_noisy for r in plateau)

    def test_ci95_shrinks_with_more_seeds(self):
        def probe(a, s):
            return float(((s * 13) % 11) - 5)
        small = empirical_alpha_sweep(probe, alphas=[1.0], seeds=[0, 1, 2])
        large = empirical_alpha_sweep(probe, alphas=[1.0], seeds=list(range(20)))
        assert small[0].ci95_half_width >= large[0].ci95_half_width

    def test_to_dict_roundtrip(self):
        sweep = empirical_alpha_sweep(
            lambda a, s: a + 0.01 * s, alphas=[0.5, 1.0], seeds=[0, 1, 2]
        )
        d = sweep[0].to_dict()
        assert set(d.keys()) >= {
            "alpha", "mean_log_margin", "std_log_margin",
            "ci95_half_width", "seed_values", "n_seeds", "is_noisy",
        }


class TestRecommend:
    def test_returns_smallest_qualified(self):
        probe = make_x7nl_like_probe()
        sweep = empirical_alpha_sweep(
            probe, alphas=[0.25, 0.7, 1.0, 1.5], seeds=[0, 1, 2, 3, 4],
        )
        rec = recommend_alpha_from_sweep(sweep, margin_floor=-1.0, noise_std_budget=1.0)
        assert rec == 0.7

    def test_fallback_when_no_qualified(self):
        sweep = [
            AlphaProbeResult(
                alpha=a, mean_log_margin=-10.0, std_log_margin=0.1,
                ci95_half_width=0.1, seed_values=[-10.0, -10.0, -10.0],
                n_seeds=3, is_noisy=False,
            )
            for a in [0.5, 1.0]
        ]
        rec = recommend_alpha_from_sweep(sweep, margin_floor=0.0, fallback=1.5)
        assert rec == 1.5


class TestFromSweep:
    def test_builds_floor_from_data(self):
        probe = make_x7nl_like_probe()
        sweep = empirical_alpha_sweep(
            probe, alphas=[0.25, 0.7, 1.0], seeds=[0, 1, 2, 3, 4],
        )
        sched = SafeAlphaScheduler.from_empirical_sweep(sweep, margin_floor=-1.0)
        assert sched.deploy_floor >= 0.5


class TestLegacyShims:
    def test_legacy_constants_are_noop(self):
        assert CLIFF_LO_DEFAULT == 0.0
        assert CLIFF_HI_DEFAULT == 0.0
        assert math.isinf(CLIFF_THRESHOLD_DEFAULT)
        assert POST_CLIFF_ALPHA_DEFAULT == RECOMMENDED_ALPHA_DEFAULT

    def test_compute_safe_alpha_threshold_emits_deprecation(self):
        with pytest.warns(DeprecationWarning, match="bf16"):
            out = compute_safe_alpha_threshold()
        assert out["deprecated"] is True
        assert out["recommended_alpha"] == RECOMMENDED_ALPHA_DEFAULT
        assert out["cliff_detected"] is False

    def test_validate_scheduler_vs_naive(self):
        sched = SafeAlphaScheduler(deploy_floor=0.7, recommended=1.0)

        def probe(a):
            if a == 0.0:
                return 0.0
            if a < 0.7:
                return -2.0
            return 0.4
        out = validate_scheduler_vs_naive(
            sched, probe, n_steps=11, log_margin_floor=-1.0,
        )
        assert out["safe_cliff_hits"] == 0
        assert out["naive_cliff_hits"] >= 4
        assert out["scheduler_passes_floor"] is True


def test_defaults_reflect_x7nl_findings():
    assert DEPLOY_ALPHA_FLOOR_DEFAULT == 0.7
    assert RECOMMENDED_ALPHA_DEFAULT == 1.0
    assert NOISE_STD_BUDGET_DEFAULT == 1.0
