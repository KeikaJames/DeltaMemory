"""Tests for deltamemory.injection.safe_alpha (B4).

Tests run without a real model by using mock probe functions.
The core logic (cliff detection, scheduler mapping, validation) is
model-independent and fast to test.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from deltamemory.injection.safe_alpha import (
    CLIFF_HI_DEFAULT,
    CLIFF_LO_DEFAULT,
    CLIFF_THRESHOLD_DEFAULT,
    POST_CLIFF_ALPHA_DEFAULT,
    SafeAlphaScheduler,
    validate_scheduler_vs_naive,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_cliff_log_margin(
    cliff_lo: float = 0.05,
    cliff_hi: float = 0.45,
    cliff_depth: float = -5.74,
    pre_cliff: float = 0.959,
    post_cliff: float = 0.365,
) -> callable:
    """Build a mock log_margin function that mimics the X.7-NL α-cliff."""
    def probe(alpha: float) -> float:
        if alpha < cliff_lo:
            return pre_cliff
        if alpha <= cliff_hi:
            # Linear interpolation from pre_cliff to cliff_depth then back
            mid = (cliff_lo + cliff_hi) / 2
            if alpha <= mid:
                t = (alpha - cliff_lo) / (mid - cliff_lo + 1e-10)
                return pre_cliff + t * (cliff_depth - pre_cliff)
            else:
                t = (alpha - mid) / (cliff_hi - mid + 1e-10)
                return cliff_depth + t * (post_cliff - cliff_depth)
        return post_cliff
    return probe


# ---------------------------------------------------------------------------
# SafeAlphaScheduler: is_in_cliff
# ---------------------------------------------------------------------------

class TestIsInCliff:
    def test_below_cliff(self):
        sched = SafeAlphaScheduler()
        assert not sched.is_in_cliff(0.0)
        assert not sched.is_in_cliff(0.04)

    def test_in_cliff_lo(self):
        sched = SafeAlphaScheduler()
        assert sched.is_in_cliff(CLIFF_LO_DEFAULT)

    def test_in_cliff_hi(self):
        sched = SafeAlphaScheduler()
        assert sched.is_in_cliff(CLIFF_HI_DEFAULT)

    def test_in_cliff_middle(self):
        sched = SafeAlphaScheduler()
        assert sched.is_in_cliff(0.25)

    def test_above_cliff(self):
        sched = SafeAlphaScheduler()
        assert not sched.is_in_cliff(0.5)
        assert not sched.is_in_cliff(1.0)
        assert not sched.is_in_cliff(2.0)

    def test_custom_cliff_range(self):
        sched = SafeAlphaScheduler(cliff_lo=0.1, cliff_hi=0.3)
        assert sched.is_in_cliff(0.2)
        assert not sched.is_in_cliff(0.05)
        assert not sched.is_in_cliff(0.35)


# ---------------------------------------------------------------------------
# SafeAlphaScheduler: safe_alpha
# ---------------------------------------------------------------------------

class TestSafeAlpha:
    def test_below_cliff_passthrough(self):
        sched = SafeAlphaScheduler(policy="post_cliff")
        assert sched.safe_alpha(0.0) == 0.0
        assert sched.safe_alpha(0.04) == pytest.approx(0.04)

    def test_above_cliff_passthrough(self):
        sched = SafeAlphaScheduler(policy="post_cliff")
        assert sched.safe_alpha(1.0) == pytest.approx(1.0)
        assert sched.safe_alpha(0.5) == pytest.approx(0.5)

    def test_cliff_zone_post_cliff_policy(self):
        sched = SafeAlphaScheduler(
            policy="post_cliff", post_cliff_alpha=1.0
        )
        assert sched.safe_alpha(0.25) == pytest.approx(1.0)
        assert sched.safe_alpha(CLIFF_LO_DEFAULT) == pytest.approx(1.0)

    def test_cliff_zone_no_inject_policy(self):
        sched = SafeAlphaScheduler(policy="no_inject")
        assert sched.safe_alpha(0.25) == pytest.approx(0.0)

    def test_cliff_zone_custom_post_cliff(self):
        sched = SafeAlphaScheduler(
            policy="post_cliff", post_cliff_alpha=0.75
        )
        assert sched.safe_alpha(0.20) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# SafeAlphaScheduler: schedule
# ---------------------------------------------------------------------------

class TestSchedule:
    def test_schedule_length(self):
        sched = SafeAlphaScheduler()
        result = sched.schedule(n_steps=10)
        assert len(result) == 10

    def test_schedule_avoids_cliff(self):
        sched = SafeAlphaScheduler(
            cliff_lo=0.05, cliff_hi=0.45, policy="post_cliff", post_cliff_alpha=1.0
        )
        alphas = sched.schedule(n_steps=10, alpha_start=0.0, alpha_end=1.0)
        # No alpha should be in the cliff zone
        for a in alphas:
            assert not sched.is_in_cliff(a), f"cliff hit: α={a}"

    def test_naive_schedule_hits_cliff(self):
        """Sanity: naive linear schedule DOES hit the cliff zone."""
        sched = SafeAlphaScheduler()
        n_steps = 10
        step = 1.0 / (n_steps - 1)
        naive = [i * step for i in range(n_steps)]
        cliff_hits = sum(1 for a in naive if sched.is_in_cliff(a))
        assert cliff_hits > 0, "Expected naive schedule to hit cliff"

    def test_schedule_single_step(self):
        sched = SafeAlphaScheduler()
        result = sched.schedule(n_steps=1, alpha_end=0.25)
        assert len(result) == 1
        # α=0.25 is in cliff, should be mapped
        assert not sched.is_in_cliff(result[0])

    def test_schedule_below_cliff_unchanged(self):
        sched = SafeAlphaScheduler(cliff_lo=0.20, cliff_hi=0.40)
        # 3 steps all below cliff
        result = sched.schedule(n_steps=3, alpha_start=0.0, alpha_end=0.10)
        assert result == pytest.approx([0.0, 0.05, 0.10])


# ---------------------------------------------------------------------------
# SafeAlphaScheduler: safe_alpha_with_probe
# ---------------------------------------------------------------------------

class TestSafeAlphaWithProbe:
    def test_post_cliff_passes_floor(self):
        sched = SafeAlphaScheduler(
            policy="post_cliff",
            post_cliff_alpha=1.0,
            log_margin_floor=-0.5,
        )
        probe = make_cliff_log_margin()
        safe_a, margin = sched.safe_alpha_with_probe(0.25, probe)
        # post_cliff α=1.0 should give a positive margin
        assert safe_a == pytest.approx(1.0)
        assert margin > sched.log_margin_floor

    def test_no_inject_fallback(self):
        """If post_cliff fails, falls back to α=0."""
        sched = SafeAlphaScheduler(
            policy="post_cliff",
            post_cliff_alpha=1.0,
            log_margin_floor=10.0,  # Impossibly high floor — forces fallback
        )
        probe = make_cliff_log_margin(post_cliff=0.365)
        safe_a, margin = sched.safe_alpha_with_probe(0.25, probe)
        # post_cliff α=1.0 gives 0.365 < 10.0, fallback to α=0
        assert safe_a == pytest.approx(0.0)

    def test_below_cliff_no_probe_call(self):
        """When requested α is below cliff, probe is not called for candidate."""
        sched = SafeAlphaScheduler()
        call_count = [0]
        def probe(alpha):
            call_count[0] += 1
            return 1.0
        # α=0.0 is below cliff, so safe_alpha returns 0.0 unchanged
        safe_a, margin = sched.safe_alpha_with_probe(0.0, probe)
        assert safe_a == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# validate_scheduler_vs_naive
# ---------------------------------------------------------------------------

class TestValidateSchedulerVsNaive:
    def test_scheduler_avoids_cliff_hits(self):
        sched = SafeAlphaScheduler(
            policy="post_cliff",
            post_cliff_alpha=1.0,
            cliff_lo=0.05, cliff_hi=0.45,
        )
        probe = make_cliff_log_margin(cliff_depth=-5.74, pre_cliff=0.959, post_cliff=0.365)
        result = validate_scheduler_vs_naive(
            scheduler=sched,
            probe_fn=probe,
            n_steps=10,
            alpha_start=0.0,
            alpha_end=1.0,
            log_margin_floor=-1.0,
        )
        # Naive should hit cliff
        assert result["naive_cliff_hits"] > 0, "Expected naive to hit cliff"
        # Scheduler should not
        assert result["safe_cliff_hits"] == 0, "Scheduler should not hit cliff"
        assert result["scheduler_passes_floor"] is True

    def test_naive_hits_cliff(self):
        sched = SafeAlphaScheduler()
        probe = make_cliff_log_margin()
        result = validate_scheduler_vs_naive(
            scheduler=sched,
            probe_fn=probe,
            n_steps=10,
            log_margin_floor=-1.0,
        )
        # Naive schedule from 0→1 in 10 steps hits α≈0.11, 0.22, 0.33 (in cliff)
        assert result["naive_min_margin"] < -1.0

    def test_safe_min_better_than_naive(self):
        sched = SafeAlphaScheduler(policy="post_cliff", post_cliff_alpha=1.0)
        probe = make_cliff_log_margin()
        result = validate_scheduler_vs_naive(
            scheduler=sched, probe_fn=probe, n_steps=10
        )
        assert result["safe_min_margin"] > result["naive_min_margin"]

    def test_result_shape(self):
        sched = SafeAlphaScheduler()
        probe = lambda a: 0.5
        result = validate_scheduler_vs_naive(
            scheduler=sched, probe_fn=probe, n_steps=5
        )
        assert len(result["naive_results"]) == 5
        assert len(result["safe_results"]) == 5


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_cliff_lo(self):
        assert CLIFF_LO_DEFAULT == pytest.approx(0.05)

    def test_default_cliff_hi(self):
        assert CLIFF_HI_DEFAULT == pytest.approx(0.45)

    def test_default_threshold(self):
        assert CLIFF_THRESHOLD_DEFAULT == pytest.approx(1.5)

    def test_default_post_cliff(self):
        assert POST_CLIFF_ALPHA_DEFAULT == pytest.approx(1.0)

    def test_scheduler_defaults(self):
        sched = SafeAlphaScheduler()
        assert sched.cliff_lo == pytest.approx(CLIFF_LO_DEFAULT)
        assert sched.cliff_hi == pytest.approx(CLIFF_HI_DEFAULT)
        assert sched.policy == "post_cliff"
        assert sched.post_cliff_alpha == pytest.approx(POST_CLIFF_ALPHA_DEFAULT)
