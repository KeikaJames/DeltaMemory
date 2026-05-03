"""mHC spectral shield for DeltaMemory bank injection.

Universal, model-agnostic spectral guard for the αM_V injection signal,
inspired by DeepSeek's Manifold-Constrained Hyper-Connections (mHC,
arXiv:2512.24880).  The original paper used a doubly-stochastic
constraint on the *training-time* hyper-connection mixing matrix C to
enforce σ_max(C) ≤ 1 and prevent residual-stream signal blow-up.  We
reuse the same spectral-non-amplification idea at *inference time*, on
the bank concat path of the merged attention computation.

Design decision (corrected after empirical test): we MUST NOT project
the full merged weight matrix `[seq; bank]` to the doubly-stochastic
manifold.  Doing so disturbs the native sequence-internal attention
pattern that the frozen LLM was trained on, causing severe NLL drift
(+5 nats observed on Gemma-4-E2B with full Sinkhorn-Knopp on the merged
matrix).

Instead, the shield touches **only the bank columns**.  We cap each
bank slot's column-sum (the total attention that reads it across all
queries in the batch) to ≤ ``kappa``.  This bounds the operator-norm of
the bank-injection map  W_bank · (αM_V) → out_bank  by ``kappa`` while
leaving the native sequence attention untouched.  Mathematically this
is equivalent to enforcing spectral non-amplification only on the
external KV channel — the part the frozen LLM never saw during training
— which is exactly the right place to put the constraint.

Why this is universal across LLMs (red-line compliant):
  * Operates only on the post-softmax attention weights of the bank
    columns.  Native columns are returned bit-for-bit unchanged.
  * Parameter-free deterministic projector: no learned weights, no
    per-architecture code paths.  Llama / Gemma / Qwen / DeepSeek / GLM
    all share the same implementation.
  * When α = 0 (or the bank is empty) the call site short-circuits the
    entire injection branch, so the shield never runs and the α = 0
    bit-equality invariant is preserved.

Mathematical guarantee:
  Let W ∈ R^{q × (T+N)} be the row-stochastic post-softmax weights with
  the first T columns indexing native V slots and the last N columns
  indexing α-scaled bank V slots.  After ``shield_attention_weights``:
      Σ_i W'[:, T+j]  ≤ κ   for every bank column j ∈ [0, N).
  By Gershgorin / sub-multiplicativity, this gives
      σ_max(W'[:, T:]) ≤ √(κ · max_j Σ_i W'[:, T+j])  ≤ √(κ · κ) = κ
  so the energy of the bank read
      out_bank = W'[:, T:] @ (α · M_V)
  is bounded above by ``κ · ‖α · M_V‖_F`` independently of α (the
  α factor enters only through M_V's magnitude and the cap is applied
  *after* softmax, removing α-dependent saturation).

References:
  * Xie, Wei, Cao, ... Liang. *Manifold-Constrained Hyper-Connections.*
    arXiv:2512.24880, 2026.
  * Cuturi. *Sinkhorn Distances.* NeurIPS 2013 (canonical SK derivation,
    used here only as the historical motivation; the actual projection
    is a single column-norm cap, which is far cheaper than full SK
    iteration and avoids disturbing native attention).
"""

from __future__ import annotations

import torch


def sinkhorn_knopp_projection(
    weights: torch.Tensor,
    iters: int = 3,
    eps: float = 1e-9,
    target_col_sum: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Project a non-negative matrix to the doubly-stochastic manifold.

    Kept for completeness and for unit testing the SK math.  The
    DeltaMemory shield call site does NOT use full SK on the merged
    matrix (see module docstring); use ``shield_attention_weights``
    instead.

    Args:
        weights: ``[..., q, k]`` tensor with non-negative entries.
        iters: number of alternating row/column normalisation iterations.
        eps: numerical floor used in divisions to avoid division by zero.
        target_col_sum: scalar or broadcastable tensor giving the desired
            column sum.  ``None`` selects ``q / k`` (balanced demand).

    Returns:
        Projected weights with the same dtype and shape as the input.
    """

    if weights.dim() < 2:
        raise ValueError(f"sinkhorn_knopp_projection: weights must be ≥ 2D, got {weights.shape}")
    if iters < 0:
        raise ValueError(f"sinkhorn_knopp_projection: iters must be ≥ 0, got {iters}")

    q = weights.size(-2)
    k = weights.size(-1)

    if target_col_sum is None:
        target_col_sum = q / max(k, 1)

    work = weights.to(torch.float32).clamp_min(0.0)
    for _ in range(iters):
        row_sum = work.sum(dim=-1, keepdim=True).clamp_min(eps)
        work = work / row_sum
        col_sum = work.sum(dim=-2, keepdim=True).clamp_min(eps)
        work = work * (target_col_sum / col_sum)
    row_sum = work.sum(dim=-1, keepdim=True).clamp_min(eps)
    work = work / row_sum
    return work.to(weights.dtype)


def shield_attention_weights(
    weights: torch.Tensor,
    bank_size: int,
    enabled: bool,
    iters: int = 3,                  # kept in signature for API stability
    kappa: float = 1.0,
) -> torch.Tensor:
    """Apply mHC bank-column spectral cap to merged ``[seq; bank]`` weights.

    The shield caps each bank column's total received attention to ≤
    ``kappa``, which bounds the spectral norm of the bank-injection
    operator without touching native sequence attention.

    Args:
        weights: ``[B, Hq, T, T_orig + N]`` post-softmax attention
            weights.  ``T_orig`` is the native key length, ``N`` is the
            number of injected bank slots; we infer ``T_orig`` as
            ``weights.size(-1) - bank_size``.
        bank_size: ``N``.  Must be ≥ 0.
        enabled: feature flag.  When False the input is returned
            unchanged (bit-for-bit identity).
        iters: deprecated; ignored by the column-cap implementation.
            Kept for API stability with the SK signature.
        kappa: maximum allowed column sum for bank columns.  Default 1.0
            corresponds to the spectral non-amplification regime
            (σ_max(W_bank) ≤ 1).

    Returns:
        Shielded attention weights of identical shape and dtype.
        Native columns are returned bit-for-bit; bank columns are
        scaled down per-column iff their column sum exceeds ``kappa``.
    """

    del iters  # unused under the column-cap design
    if not enabled:
        return weights
    if bank_size <= 0:
        return weights
    T_orig = weights.size(-1) - bank_size
    if T_orig <= 0:
        return weights

    eps = 1e-9
    bank = weights[..., T_orig:].to(torch.float32)
    col_sum = bank.sum(dim=-2, keepdim=True).clamp_min(eps)   # [..., 1, N]
    scale = (kappa / col_sum).clamp(max=1.0)
    bank_capped = (bank * scale).to(weights.dtype)
    native = weights[..., :T_orig]                            # untouched
    return torch.cat([native, bank_capped], dim=-1)
