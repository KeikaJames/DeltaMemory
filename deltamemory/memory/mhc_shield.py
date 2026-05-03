"""mHC spectral shield for DeltaMemory bank injection.

Universal, model-agnostic spectral guard for the αM_V injection signal,
inspired by DeepSeek's Manifold-Constrained Hyper-Connections (mHC,
arXiv:2512.24880).  The original paper used Sinkhorn-Knopp projection on
the *training-time* hyper-connection mixing matrix C to enforce
σ_max(C) ≤ 1 and prevent residual-stream signal blow-up.  We reuse the
same mathematical machinery at *inference time*, on the merged
attention-weight matrix that combines native sequence tokens with the
external DeltaMemory bank slots.

Why this is universal across LLMs (red-line compliant):
  * The shield operates ONLY on the post-softmax attention weights of the
    DeltaMemory bank concat path (``[V; αM_V]``).  It does NOT touch the
    LLM's W_q / W_k / W_v / W_o weights, nor any FFN, nor any layernorm.
  * It is a parameter-free deterministic projector: no learned weights,
    no per-architecture code paths.  Llama / Gemma / Qwen / DeepSeek /
    GLM all use the exact same implementation.
  * When α = 0 (or the bank is empty) the caller must skip the shield
    entirely; in that case the standard softmax weights are returned
    bit-for-bit, so the α = 0 conservation invariant is preserved.

Mathematical guarantee (informal):
  Let W ∈ R^{q × (T+N)} be the row-stochastic post-softmax weights with
  the first T columns indexing native V slots and the last N columns
  indexing α-scaled bank V slots.  Sinkhorn-Knopp produces W* with
  Σ_j W*[i,j] = 1 ∀i  and  Σ_i W*[i,j] = (q / (T+N)) ∀j (column-balanced
  to the "uniform demand" factor q/(T+N); for non-square matrices the
  row/column sums target the doubly-stochastic-with-marginals projection).
  In particular, σ_max(W*) ≤ 1, so the energy of the merged read
      out  =  W*[:, :T] @ V  +  W*[:, T:] @ (α · M_V)
  is bounded above by the energy of the input matrix [V; αM_V] with the
  same spectral norm — independently of the chosen α scale.

References:
  * Xie, Wei, Cao, ... Liang. *Manifold-Constrained Hyper-Connections.*
    arXiv:2512.24880, 2026.
  * Cuturi. *Sinkhorn Distances.* NeurIPS 2013 (canonical SK derivation).
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

    Args:
        weights: ``[..., q, k]`` tensor with non-negative entries (typically
            post-softmax attention weights).  The leading ``...`` dims (e.g.
            batch / head) are treated independently.
        iters: number of alternating row/column normalisation iterations.
            Three is sufficient to reduce row/column-sum L1 deviation
            below 1e-3 for typical attention matrices.
        eps: numerical floor used in divisions to avoid division by zero.
        target_col_sum: scalar or broadcastable tensor giving the desired
            column sum.  For ``q != k`` the natural choice is ``q / k``
            (balanced demand).  ``None`` selects this default.

    Returns:
        Projected weights with the same dtype and shape as the input.  The
        result satisfies ``Σ_j W[..., i, j] ≈ 1`` for every row and
        ``Σ_i W[..., i, j] ≈ target_col_sum`` for every column to within
        the chosen iteration tolerance.
    """

    if weights.dim() < 2:
        raise ValueError(f"sinkhorn_knopp_projection: weights must be ≥ 2D, got {weights.shape}")
    if iters < 0:
        raise ValueError(f"sinkhorn_knopp_projection: iters must be ≥ 0, got {iters}")

    q = weights.size(-2)
    k = weights.size(-1)

    if target_col_sum is None:
        target_col_sum = q / max(k, 1)

    work = weights.to(torch.float32)
    work = work.clamp_min(0.0)

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
    iters: int = 3,
) -> torch.Tensor:
    """Apply mHC spectral shield to merged ``[seq; bank]`` attention weights.

    This is the high-level entry point used by ``attn_native_bank``.  It
    is a no-op when ``enabled`` is False or when the bank is empty, so
    the α = 0 conservation invariant is preserved by short-circuiting at
    the call site rather than relying on Sinkhorn fixed-point arithmetic.

    Args:
        weights: ``[B, Hq, T, T_orig + N]`` post-softmax attention
            weights.  ``T_orig`` is the native sequence length (or the
            attention key length if causal masking shortens it),
            ``N`` is the number of injected bank slots.
        bank_size: ``N``.  Must be ≥ 0.
        enabled: feature flag.  When False the input is returned
            unchanged (bit-for-bit identity).
        iters: number of Sinkhorn-Knopp iterations.  Default 3.

    Returns:
        Shielded attention weights of identical shape and dtype.
    """

    if not enabled:
        return weights
    if bank_size <= 0:
        return weights
    if weights.size(-1) <= bank_size:
        # No native slots present (degenerate case); SK projection on a
        # single block reduces to the identity column scaling.  Skip.
        return weights

    return sinkhorn_knopp_projection(weights, iters=iters)
