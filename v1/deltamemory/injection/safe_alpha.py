"""Empirical α-sweep probe + deployable α scheduler for AttnNativeBank.

REWRITTEN 2026-05-06 (v0.7): the original v0.6 implementation modelled an
"α=0.25 cliff" / "phase transition" on gemma-4-31B-it. That model was
based on **single-seed** values from the X.7-NL subB sweep. A
multi-seed analysis (`runs/X7NL_full_v1_gemma4_31B/ALPHA_CLIFF_DEBUNK.md`)
showed that the apparent cliff is bf16 + seed noise:

    α range          per-seed std of log_margin
    [0.05, 0.50]     ~1.70 nats   (noisy bowl)
    [0.70, 2.00]     ~0.31 nats   (stable plateau)

α only enters the bank-V output as a multiplicative scalar; it is not
inside any softmax. There is no mathematical mechanism by which α=0.25
could be a phase transition.

This module therefore now exposes:

* :func:`empirical_alpha_sweep` — runs an honest per-seed sweep and
  reports mean / std / 95% CI per α, plus a "deployable α" floor.
* :class:`SafeAlphaScheduler` — a no-op-by-default mapper that can
  optionally enforce a deploy-α floor (default 0.7) to stay out of the
  high-variance region.
* :func:`recommend_alpha_from_sweep` — pick the smallest α whose mean
  log_margin clears a floor AND whose per-seed std is below a noise
  budget.

The public API names are kept stable for backward compatibility with
PR #38 and downstream code; the **semantics** are now empirical, not
phantom-based.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import torch  # noqa: F401  (used by callers; keeps import surface stable)


# ---------------------------------------------------------------------------
# Defaults — derived from real X.7-NL gemma-4-31B-it 3-seed data
# ---------------------------------------------------------------------------

DEPLOY_ALPHA_FLOOR_DEFAULT: float = 0.7
RECOMMENDED_ALPHA_DEFAULT: float = 1.0
NOISE_STD_BUDGET_DEFAULT: float = 1.0
EMPIRICAL_SWEEP_SEEDS_DEFAULT: tuple[int, ...] = (0, 1, 2)


# ---------------------------------------------------------------------------
# Empirical sweep
# ---------------------------------------------------------------------------

@dataclass
class AlphaProbeResult:
    """Per-α statistics from an empirical sweep."""
    alpha: float
    mean_log_margin: float
    std_log_margin: float
    ci95_half_width: float
    seed_values: list[float]
    n_seeds: int
    is_noisy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "mean_log_margin": self.mean_log_margin,
            "std_log_margin": self.std_log_margin,
            "ci95_half_width": self.ci95_half_width,
            "seed_values": list(self.seed_values),
            "n_seeds": self.n_seeds,
            "is_noisy": self.is_noisy,
        }


def empirical_alpha_sweep(
    probe_fn: Callable[[float, int], float],
    *,
    alphas: Optional[Iterable[float]] = None,
    seeds: Iterable[int] = EMPIRICAL_SWEEP_SEEDS_DEFAULT,
    noise_std_budget: float = NOISE_STD_BUDGET_DEFAULT,
) -> list[AlphaProbeResult]:
    """Run a real per-seed α sweep using user-supplied probe_fn."""
    if alphas is None:
        alphas = (0.0, 0.25, 0.5, 0.7, 1.0, 1.5, 2.0)
    seed_list = list(seeds)
    if len(seed_list) < 2:
        raise ValueError(
            f"empirical_alpha_sweep requires ≥2 seeds (got {len(seed_list)}); "
            "without multi-seed measurement we cannot detect noise."
        )

    out: list[AlphaProbeResult] = []
    for alpha in alphas:
        per_seed: list[float] = []
        for s in seed_list:
            v = float(probe_fn(alpha, s))
            if math.isfinite(v):
                per_seed.append(v)
        if not per_seed:
            continue
        mean = sum(per_seed) / len(per_seed)
        if len(per_seed) >= 2:
            var = sum((v - mean) ** 2 for v in per_seed) / (len(per_seed) - 1)
            std = math.sqrt(var)
        else:
            std = float("nan")
        ci95 = 1.96 * std / math.sqrt(len(per_seed)) if std == std else float("nan")
        out.append(
            AlphaProbeResult(
                alpha=float(alpha),
                mean_log_margin=mean,
                std_log_margin=std,
                ci95_half_width=ci95,
                seed_values=per_seed,
                n_seeds=len(per_seed),
                is_noisy=bool(std > noise_std_budget),
            )
        )
    return out


def recommend_alpha_from_sweep(
    sweep: list[AlphaProbeResult],
    *,
    margin_floor: float = 0.0,
    noise_std_budget: float = NOISE_STD_BUDGET_DEFAULT,
    fallback: float = RECOMMENDED_ALPHA_DEFAULT,
) -> float:
    """Pick the smallest α with mean ≥ floor AND std ≤ budget."""
    candidates = [
        r for r in sweep
        if r.alpha > 0
        and r.mean_log_margin >= margin_floor
        and r.std_log_margin <= noise_std_budget
    ]
    if not candidates:
        return fallback
    return min(c.alpha for c in candidates)


# ---------------------------------------------------------------------------
# SafeAlphaScheduler — backward-compatible API, empirical defaults
# ---------------------------------------------------------------------------


@dataclass
class SafeAlphaScheduler:
    """Map requested α to a deployable α (empirical noise-floor based)."""

    deploy_floor: float = DEPLOY_ALPHA_FLOOR_DEFAULT
    recommended: float = RECOMMENDED_ALPHA_DEFAULT
    allow_zero: bool = True

    def is_safe(self, alpha: float) -> bool:
        if alpha == 0.0 and self.allow_zero:
            return True
        return alpha >= self.deploy_floor

    def safe_alpha(self, requested_alpha: float) -> float:
        if self.is_safe(requested_alpha):
            return requested_alpha
        return self.recommended

    def schedule(
        self,
        n_steps: int,
        alpha_start: float = 0.0,
        alpha_end: float = 1.0,
    ) -> list[float]:
        if n_steps <= 1:
            return [self.safe_alpha(alpha_end)]
        step = (alpha_end - alpha_start) / (n_steps - 1)
        naive = [alpha_start + i * step for i in range(n_steps)]
        return [self.safe_alpha(a) for a in naive]

    @classmethod
    def from_empirical_sweep(
        cls,
        sweep: list[AlphaProbeResult],
        *,
        margin_floor: float = 0.0,
        noise_std_budget: float = NOISE_STD_BUDGET_DEFAULT,
    ) -> "SafeAlphaScheduler":
        recommended = recommend_alpha_from_sweep(
            sweep,
            margin_floor=margin_floor,
            noise_std_budget=noise_std_budget,
        )
        return cls(
            deploy_floor=recommended,
            recommended=recommended,
        )


# ---------------------------------------------------------------------------
# Deprecated legacy aliases (PR #38 import surface compatibility).
# ---------------------------------------------------------------------------

CLIFF_LO_DEFAULT: float = 0.0
CLIFF_HI_DEFAULT: float = 0.0
CLIFF_THRESHOLD_DEFAULT: float = float("inf")
POST_CLIFF_ALPHA_DEFAULT: float = RECOMMENDED_ALPHA_DEFAULT


def compute_safe_alpha_threshold(
    *_args: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Deprecated. Returns a stub recommending α=1.0."""
    warnings.warn(
        "compute_safe_alpha_threshold is deprecated; the α=0.25 cliff it "
        "modelled was bf16+seed noise. Use empirical_alpha_sweep() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {
        "cliff_detected": False,
        "cliff_alpha": None,
        "l_cliff_detected": None,
        "largest_safe_alpha": RECOMMENDED_ALPHA_DEFAULT,
        "smallest_recovery_alpha": RECOMMENDED_ALPHA_DEFAULT,
        "recommended_alpha": RECOMMENDED_ALPHA_DEFAULT,
        "probe_results": [],
        "deprecated": True,
        "note": (
            "α=0.25 cliff is bf16+seed noise, not a phase transition. "
            "See runs/X7NL_full_v1_gemma4_31B/ALPHA_CLIFF_DEBUNK.md."
        ),
    }


def validate_scheduler_vs_naive(
    scheduler: "SafeAlphaScheduler",
    probe_fn: Callable[[float], float],
    n_steps: int = 10,
    alpha_start: float = 0.0,
    alpha_end: float = 1.0,
    log_margin_floor: float = -1.0,
) -> dict[str, Any]:
    """Compare naive linear schedule vs SafeAlphaScheduler (empirical)."""
    step = (alpha_end - alpha_start) / max(n_steps - 1, 1)
    naive_alphas = [alpha_start + i * step for i in range(n_steps)]
    safe_alphas = scheduler.schedule(n_steps, alpha_start, alpha_end)

    naive_results = [
        {"alpha": a, "log_margin": float(probe_fn(a))} for a in naive_alphas
    ]
    safe_results = []
    for a_naive, a_safe in zip(naive_alphas, safe_alphas):
        if a_safe == a_naive:
            margin = next(r["log_margin"] for r in naive_results if r["alpha"] == a_naive)
        else:
            margin = float(probe_fn(a_safe))
        safe_results.append({"alpha": a_safe, "log_margin": margin})

    naive_min = min(r["log_margin"] for r in naive_results)
    safe_min = min(r["log_margin"] for r in safe_results)
    naive_floor_hits = sum(1 for r in naive_results if r["log_margin"] < log_margin_floor)
    safe_floor_hits = sum(1 for r in safe_results if r["log_margin"] < log_margin_floor)

    return {
        "naive_min_margin": naive_min,
        "safe_min_margin": safe_min,
        "naive_cliff_hits": naive_floor_hits,
        "safe_cliff_hits": safe_floor_hits,
        "scheduler_passes_floor": bool(safe_floor_hits == 0),
        "naive_results": naive_results,
        "safe_results": safe_results,
    }


__all__ = [
    "AlphaProbeResult",
    "empirical_alpha_sweep",
    "recommend_alpha_from_sweep",
    "SafeAlphaScheduler",
    "compute_safe_alpha_threshold",
    "validate_scheduler_vs_naive",
    "DEPLOY_ALPHA_FLOOR_DEFAULT",
    "RECOMMENDED_ALPHA_DEFAULT",
    "NOISE_STD_BUDGET_DEFAULT",
    "CLIFF_LO_DEFAULT",
    "CLIFF_HI_DEFAULT",
    "CLIFF_THRESHOLD_DEFAULT",
    "POST_CLIFF_ALPHA_DEFAULT",
]
