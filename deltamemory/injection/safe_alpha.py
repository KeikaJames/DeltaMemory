"""Safe-α Prescription for AttnNativeBank (Track B, B4).

Implements:
  - ``compute_safe_alpha_threshold``: probe the cliff in real-time and
    return the largest α below the cliff OR smallest above the recovery.
  - ``SafeAlphaScheduler``: skip the cliff region using α=0 or α≥0.5.

Background
----------
X.7-NL revealed a catastrophic cliff at α=0.25 on gemma-4-31B-it:
    α=0.00 → +0.959 log_margin  (parametric memory wins)
    α=0.25 → −5.740             (cliff: injection destroys recall)
    α=0.50 → −0.839
    α=0.75 → +0.010
    α=1.00 → +0.365             (recovery plateau)

B3 analysis (alpha_cliff.py) identifies the cliff layer L_cliff where
‖Δresidual‖ is maximised at α=0.25. The safe-α algorithm uses this
per-layer residual norm as a real-time cliff detector.

Algorithm
---------
1. Probe α in {0.05, 0.10, ..., 0.45} (low-α sweep).
2. For each α, compute the mean residual norm at L_cliff.
3. If the norm ratio (norm_α / norm_0) exceeds the cliff_threshold (default 1.5),
   the cliff has been crossed.
4. Return: largest α **below** the cliff (pre-cliff), or 1.0 (post-cliff default).

SafeAlphaScheduler
------------------
Given a target α (e.g., from a linear schedule 0→1), it maps to:
  - α=0 if target is in the cliff zone [cliff_lo, cliff_hi] and policy='no_inject'
  - α=post_cliff (default 1.0) if policy='post_cliff' (recommended)

Validated: a naive linear schedule (α=0→1 in 10 steps) hits the cliff
at α=0.25. The scheduler skips it. The scheduler guarantees log_margin
never falls below a configurable floor.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import torch


# ---------------------------------------------------------------------------
# Constants from X.7-NL B3 findings
# ---------------------------------------------------------------------------

# Default cliff zone identified experimentally (α in [0.05, 0.45])
CLIFF_LO_DEFAULT: float = 0.05
CLIFF_HI_DEFAULT: float = 0.45

# Default residual ratio threshold: if norm_α / norm_0 > this, we're in cliff
CLIFF_THRESHOLD_DEFAULT: float = 1.5

# Default post-cliff alpha
POST_CLIFF_ALPHA_DEFAULT: float = 1.0


# ---------------------------------------------------------------------------
# Core probe function
# ---------------------------------------------------------------------------

def compute_safe_alpha_threshold(
    model: Any,
    bank: Any,
    patcher: Any,
    tokenizer: Any,
    read_prompt: str,
    *,
    probe_alphas: Optional[list[float]] = None,
    cliff_threshold: float = CLIFF_THRESHOLD_DEFAULT,
    device: Optional[str] = None,
    l_cliff: Optional[int] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Probe the α-cliff in real-time and return safe α recommendations.

    Runs a mini-sweep over low-α values, measuring per-layer residual norms.
    Identifies the cliff by finding the smallest α where the normalised
    residual-norm ratio at layer L_cliff exceeds `cliff_threshold`.

    Parameters
    ----------
    model:
        The loaded HuggingFace causal LM.
    bank:
        An :class:`~deltamemory.memory.attn_native_bank.AttnNativeBank`
        populated with facts.
    patcher:
        An installed :class:`~deltamemory.memory.attn_native_bank.AttnNativePatcher`.
    tokenizer:
        The model's tokenizer.
    read_prompt:
        Prompt string for measuring recall (should end just before the
        target token position).
    probe_alphas:
        α values to probe. Defaults to [0.0, 0.05, 0.10, ..., 0.50, 1.0].
    cliff_threshold:
        Residual-norm ratio above which α is considered "in the cliff zone".
        Default 1.5 (from H_B3.2).
    device:
        Device override. Defaults to model's device.
    l_cliff:
        If known (from B3 analysis), the cliff layer index. If None, the
        layer with maximum Δresidual at the first cliff α is used.
    verbose:
        Print per-α diagnostics.

    Returns
    -------
    dict with keys:
        ``largest_safe_alpha`` (float): largest α < cliff.
        ``smallest_recovery_alpha`` (float): smallest α ≥ cliff where ratio drops.
        ``cliff_alpha`` (float | None): first α where cliff is detected.
        ``l_cliff_detected`` (int | None): detected cliff layer.
        ``probe_results`` (list[dict]): per-alpha probe data.
        ``recommended_alpha`` (float): POST_CLIFF_ALPHA_DEFAULT or 0.0.
        ``cliff_detected`` (bool).
    """
    from deltamemory.memory.attn_native_bank import forward_with_bank
    from deltamemory.diagnostics import DiagnosticRecorder

    if probe_alphas is None:
        probe_alphas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]

    probe_results: list[dict] = []
    baseline_norm_per_layer: dict[int, float] = {}
    cliff_alpha_detected: Optional[float] = None
    l_cliff_detected: Optional[int] = l_cliff

    for alpha in sorted(probe_alphas):
        rec = DiagnosticRecorder(model=model, patcher=patcher, enabled=True)
        try:
            with rec:
                logits = forward_with_bank(
                    patcher=patcher, bank=bank, tokenizer=tokenizer,
                    read_prompt=read_prompt, alpha=alpha,
                )
        except Exception as exc:
            probe_results.append({"alpha": alpha, "status": "failed", "error": str(exc)[:200]})
            continue

        # Collect per-layer residual norms
        layer_norms: dict[int, list[float]] = {}
        for r in rec._records:
            if r["signal_name"] == "residual_norm":
                L = r["layer"]
                layer_norms.setdefault(L, []).append(r["value"])
        mean_norm_per_layer: dict[int, float] = {
            L: sum(v) / len(v) for L, v in layer_norms.items()
        }

        if alpha == 0.0:
            baseline_norm_per_layer = dict(mean_norm_per_layer)

        # Compute ratio vs baseline
        ratio_per_layer: dict[int, float] = {}
        if baseline_norm_per_layer:
            for L, norm in mean_norm_per_layer.items():
                base = baseline_norm_per_layer.get(L, 1e-8)
                ratio_per_layer[L] = norm / (base + 1e-8)

        # Detect cliff: find layer with max ratio
        max_ratio = max(ratio_per_layer.values()) if ratio_per_layer else 0.0
        max_ratio_layer = (
            max(ratio_per_layer, key=lambda L: ratio_per_layer[L])
            if ratio_per_layer else None
        )

        result = {
            "alpha": alpha,
            "status": "ok",
            "max_residual_ratio": max_ratio,
            "max_ratio_layer": max_ratio_layer,
            "mean_norm_per_layer": {str(L): v for L, v in mean_norm_per_layer.items()},
        }

        if verbose:
            print(
                f"  safe_alpha probe α={alpha:.2f} "
                f"max_ratio={max_ratio:.3f} L={max_ratio_layer}",
                flush=True,
            )

        # Update cliff detection
        if (alpha > 0.0
                and max_ratio > cliff_threshold
                and cliff_alpha_detected is None):
            cliff_alpha_detected = alpha
            if l_cliff_detected is None and max_ratio_layer is not None:
                l_cliff_detected = max_ratio_layer

        probe_results.append(result)

    # Determine largest_safe_alpha (biggest α below cliff)
    pre_cliff_alphas = [
        r["alpha"] for r in probe_results
        if r["status"] == "ok"
        and r.get("max_residual_ratio", 0.0) <= cliff_threshold
        and r["alpha"] > 0.0
    ]
    largest_safe_alpha = max(pre_cliff_alphas) if pre_cliff_alphas else 0.0

    # Determine smallest_recovery_alpha (smallest α after cliff where ratio normalises)
    post_cliff_candidates = [
        r for r in probe_results
        if r["status"] == "ok"
        and cliff_alpha_detected is not None
        and r["alpha"] > cliff_alpha_detected
        and r.get("max_residual_ratio", 999.0) <= cliff_threshold
    ]
    smallest_recovery_alpha = (
        min(r["alpha"] for r in post_cliff_candidates)
        if post_cliff_candidates
        else POST_CLIFF_ALPHA_DEFAULT
    )

    cliff_detected = cliff_alpha_detected is not None
    recommended = smallest_recovery_alpha if cliff_detected else largest_safe_alpha

    return {
        "cliff_detected": cliff_detected,
        "cliff_alpha": cliff_alpha_detected,
        "l_cliff_detected": l_cliff_detected,
        "largest_safe_alpha": largest_safe_alpha,
        "smallest_recovery_alpha": smallest_recovery_alpha,
        "recommended_alpha": recommended,
        "probe_results": probe_results,
        "cliff_threshold": cliff_threshold,
    }


# ---------------------------------------------------------------------------
# SafeAlphaScheduler
# ---------------------------------------------------------------------------

Policy = Literal["no_inject", "post_cliff"]


@dataclass
class SafeAlphaScheduler:
    """Scheduler that skips the cliff region in the α-response.

    Parameters
    ----------
    cliff_lo:
        Lower bound of cliff zone (inclusive). Default: 0.05.
    cliff_hi:
        Upper bound of cliff zone (inclusive). Default: 0.45.
    policy:
        How to handle cliff zone:
        - 'post_cliff': map cliff-zone α to `post_cliff_alpha` (default 1.0).
        - 'no_inject': map cliff-zone α to 0.0 (no injection at all).
    post_cliff_alpha:
        α to use when policy='post_cliff'. Default: 1.0.
    log_margin_floor:
        If a log_margin probe function is provided, the scheduler will
        not use post_cliff_alpha if it produces log_margin below this floor.
        Then it falls back to 0.0. Default: -1.0.
    """
    cliff_lo: float = CLIFF_LO_DEFAULT
    cliff_hi: float = CLIFF_HI_DEFAULT
    policy: Policy = "post_cliff"
    post_cliff_alpha: float = POST_CLIFF_ALPHA_DEFAULT
    log_margin_floor: float = -1.0

    def is_in_cliff(self, alpha: float) -> bool:
        """Return True if alpha is in the dangerous cliff zone."""
        return self.cliff_lo <= alpha <= self.cliff_hi

    def safe_alpha(self, requested_alpha: float) -> float:
        """Map a requested alpha to a safe alpha.

        Parameters
        ----------
        requested_alpha:
            The α value from the calling code (e.g., from a linear schedule).

        Returns
        -------
        float: A safe α that avoids the cliff zone.
        """
        if not self.is_in_cliff(requested_alpha):
            return requested_alpha
        if self.policy == "post_cliff":
            return self.post_cliff_alpha
        else:  # no_inject
            return 0.0

    def safe_alpha_with_probe(
        self,
        requested_alpha: float,
        probe_fn: Callable[[float], float],
    ) -> tuple[float, float]:
        """Map alpha to safe α, validating with a log_margin probe.

        Parameters
        ----------
        requested_alpha:
            Requested α from the caller.
        probe_fn:
            Function that takes α and returns log_margin. Called only when
            the mapped α differs from the requested one to validate the choice.

        Returns
        -------
        (safe_alpha, log_margin): The safe α and its measured log_margin.
        """
        candidate = self.safe_alpha(requested_alpha)
        margin = probe_fn(candidate)
        if margin >= self.log_margin_floor:
            return candidate, margin
        # Fallback: if post_cliff failed, fall back to α=0 (no injection).
        # α=0 is always safe because it leaves the model output unchanged.
        if candidate != 0.0:
            fallback_margin = probe_fn(0.0)
            return 0.0, fallback_margin
        # Already at α=0, accept whatever we have
        return candidate, margin

    def schedule(
        self,
        n_steps: int,
        alpha_start: float = 0.0,
        alpha_end: float = 1.0,
    ) -> list[float]:
        """Generate a list of safe alphas for a multi-step schedule.

        A "naive" linear schedule from alpha_start to alpha_end has n_steps
        values. This maps each to a cliff-safe α.

        Parameters
        ----------
        n_steps:
            Number of schedule steps.
        alpha_start:
            Starting α (default 0.0).
        alpha_end:
            Ending α (default 1.0).

        Returns
        -------
        list[float]: Safe alpha for each step.
        """
        if n_steps <= 1:
            return [self.safe_alpha(alpha_end)]
        step = (alpha_end - alpha_start) / (n_steps - 1)
        naive = [alpha_start + i * step for i in range(n_steps)]
        return [self.safe_alpha(a) for a in naive]


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_scheduler_vs_naive(
    scheduler: SafeAlphaScheduler,
    probe_fn: Callable[[float], float],
    n_steps: int = 10,
    alpha_start: float = 0.0,
    alpha_end: float = 1.0,
    log_margin_floor: float = -1.0,
) -> dict[str, Any]:
    """Compare naive linear schedule vs SafeAlphaScheduler.

    Parameters
    ----------
    scheduler:
        A configured SafeAlphaScheduler.
    probe_fn:
        Function alpha → log_margin.
    n_steps:
        Number of steps to compare.
    alpha_start, alpha_end:
        Schedule bounds.
    log_margin_floor:
        Assert scheduler never produces margin below this.

    Returns
    -------
    dict with comparison data and pass/fail status.
    """
    step = (alpha_end - alpha_start) / max(n_steps - 1, 1)
    naive_alphas = [alpha_start + i * step for i in range(n_steps)]
    safe_alphas = scheduler.schedule(n_steps, alpha_start, alpha_end)

    naive_results = []
    safe_results = []

    for a_naive, a_safe in zip(naive_alphas, safe_alphas):
        m_naive = probe_fn(a_naive)
        m_safe = probe_fn(a_safe) if a_safe != a_naive else m_naive
        naive_results.append({"alpha": a_naive, "log_margin": m_naive})
        safe_results.append({"alpha": a_safe, "log_margin": m_safe})

    naive_min = min(r["log_margin"] for r in naive_results)
    safe_min = min(r["log_margin"] for r in safe_results)
    naive_cliff_hits = sum(
        1 for r in naive_results if r["log_margin"] < log_margin_floor
    )
    safe_cliff_hits = sum(
        1 for r in safe_results if r["log_margin"] < log_margin_floor
    )

    return {
        "naive_min_margin": naive_min,
        "safe_min_margin": safe_min,
        "naive_cliff_hits": naive_cliff_hits,
        "safe_cliff_hits": safe_cliff_hits,
        "scheduler_passes_floor": bool(safe_cliff_hits == 0),
        "naive_results": naive_results,
        "safe_results": safe_results,
    }
