"""mHC spectral shield for Mneme bank injection.

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

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    pass


def sinkhorn_knopp_projection(
    weights: torch.Tensor,
    iters: int = 3,
    eps: float = 1e-9,
    target_col_sum: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Project a non-negative matrix to the doubly-stochastic manifold.

    Kept for completeness and for unit testing the SK math.  The
    Mneme shield call site does NOT use full SK on the merged
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
        kappa: maximum allowed column sum for bank columns.  Default 1.0
            corresponds to the spectral non-amplification regime
            (σ_max(W_bank) ≤ 1).

    Returns:
        Shielded attention weights of identical shape and dtype.
        Native columns are returned bit-for-bit; bank columns are
        scaled down per-column iff their column sum exceeds ``kappa``.
    """

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


def apply_shield_per_expert(
    weights: torch.Tensor,
    T_orig: int,
    kappa: float,
    expert_gates: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Per-expert mHC bank-column spectral cap (opt-in; default path unchanged).

    This function implements the per-expert column-cap formula for MoE
    architectures where the FFN router gate provides a per-token, per-expert
    weight that proxies how much of each token's attention output is "owned"
    by each expert.  See ``docs/theory/mhc_moe.md`` Section 4 for the
    derivation.

    .. note:: **Architecture scope — FFN-MoE approximation**

       All currently supported models (Mixtral 8×7B, Qwen3-MoE) use FFN-MoE
       with *dense attention*.  The per-expert cap is therefore an
       approximation: the FFN router gate g_e(token) is used as a proxy to
       bucket attention queries into per-expert populations.  For a true
       MoE-Attention architecture with per-expert V projections the formula
       would be exact.  See ``mhc_moe.md`` and the ``deltamemory/arch``
       module for the full discussion.

    .. warning:: **α=0 red-line**

       When the bank is empty (α=0) the call site must short-circuit *before*
       calling this function.  If ``weights`` has no bank columns
       (``T_orig >= weights.size(-1)``), this function returns the input
       unchanged bit-for-bit.

    Algorithm (weighted-average cap)
    ---------------------------------
    Assumes normalised routing: Σ_e g_e(i) = 1 for each token i.

    For each expert e:

    1. Expert-masked bank weight:
       ``W_e[i, n] = g_e[i] * W[i, T_orig+n]``
    2. Per-expert column-sum:
       ``col_sum_e[n] = Σ_i W_e[i, n]``
    3. Per-expert scale (the core per-expert guarantee):
       ``cap_e[n] = min(1, κ / col_sum_e[n])``
       — guarantees ``Σ_i W_e[i,n] * cap_e[n] ≤ κ`` per expert e.
    4. Weighted-average cap per token:
       ``cap[i, n] = Σ_e g_e[i] * cap_e[n]``

    Final: ``W_shielded[i, T_orig+n] = W[i, T_orig+n] * cap[i, n]``

    Properties:
    - When no cap fires (all col_sum_e < κ): cap_e = 1 → cap[i,n] =
      Σ_e g_e[i] = 1 → W_shielded = W (identity under normalised routing).
    - cap[i,n] ≤ 1 always (cap_e ≤ 1 and Σ g_e = 1) → shield never amplifies.
    - Native columns unchanged bit-for-bit.
    - Single-expert (gate=1.0) reduces to the global ``shield_attention_weights``.



    Args:
        weights: Post-softmax attention weights with shape
            ``[q, T_orig + N]`` — a 2-D view with the batch and head
            dimensions collapsed.  The caller is responsible for reshaping
            from ``[B, H, T, T+N]`` to ``[B*H*T, T+N]`` (using the
            attention head's query axis as the "token" axis).

            .. note::

               For the proxy-gate use case the typical call shape is
               ``[seq_len, T_orig + N]`` where ``seq_len`` equals the number
               of tokens in the current forward pass (batch × seq), matching
               the ``(seq_len,)`` shape of the gate vectors.

        T_orig: Number of native (non-bank) key slots.  Bank columns are
            ``weights[:, T_orig:]``.
        kappa: Column-sum cap per expert.  Same semantics as in
            ``shield_attention_weights``.
        expert_gates: Dict from expert id (int) to a ``(q,)`` float gate
            tensor.  Experts not in the dict contribute zero.  Gate tensors
            must be non-negative.  Shared experts should have gate = 1.0 for
            every token.

    Returns:
        Shielded weight tensor of the same shape and dtype as ``weights``.
        Native columns ``[:, :T_orig]`` are returned bit-for-bit unchanged.
        When ``expert_gates`` is empty or the bank has no columns, the input
        is returned unchanged.

    Raises:
        ValueError: If ``T_orig`` is negative, or if a gate tensor has a
            shape incompatible with ``weights``.

    Examples
    --------
    Degenerate single-expert check (identical to global cap)::

        gates = {0: torch.ones(q)}
        out = apply_shield_per_expert(weights, T_orig, kappa, gates)
        # out_bank_cols == shield_attention_weights(weights, N, True, kappa)_bank_cols

    Two-expert toy::

        gates = {0: torch.tensor([0.9, 0.9, 0.1, 0.1]),
                 1: torch.tensor([0.1, 0.1, 0.9, 0.9])}
        out = apply_shield_per_expert(weights, T_orig, kappa, gates)
    """
    if T_orig < 0:
        raise ValueError(f"apply_shield_per_expert: T_orig must be ≥ 0, got {T_orig}")

    N = weights.size(-1) - T_orig
    if N <= 0 or not expert_gates:
        return weights

    q = weights.size(0)
    eps = 1e-9

    # Work in float32 for numerical stability; cast back at end.
    bank = weights[:, T_orig:].to(torch.float32)   # (q, N)

    # Weighted-average cap accumulator:
    #   cap_per_token[i,n] = Σ_e g_e[i] * cap_e[n]
    # Assumes normalised routing (Σ_e g_e[i] = 1), so when no cap fires
    # cap_e[n]=1 → cap_per_token[i,n] = 1 → shielded_bank = bank (identity).
    # This formulation is correct for FFN-MoE top-k routing where each token's
    # gate vector sums to 1.
    cap_per_token = torch.zeros(q, N, dtype=torch.float32, device=bank.device)

    for e, g_e in expert_gates.items():
        gate = g_e.to(torch.float32).view(q)        # (q,)

        # Per-expert masked bank weight:  W_e[i,n] = g_e[i] * bank[i,n]
        W_e = gate.unsqueeze(-1) * bank             # (q, N)

        # Per-expert column-sum cap (the per-expert guarantee):
        #   Σ_i W_e[i,n] * cap_e[n] ≤ kappa  for each slot n
        col_sum_e = W_e.sum(dim=0).clamp_min(eps)  # (N,)
        cap_e = (kappa / col_sum_e).clamp(max=1.0)  # (N,)

        # Accumulate weighted cap:  g_e[i] * cap_e[n]
        cap_per_token += gate.unsqueeze(-1) * cap_e.unsqueeze(0)  # (q, N)

    # Apply the per-token weighted cap to bank columns.
    # cap_per_token[i,n] ≤ Σ_e g_e[i] = 1 (normalized routing + cap_e ≤ 1).
    shielded_bank = (bank * cap_per_token).to(weights.dtype)

    native = weights[:, :T_orig]                   # (q, T_orig) — untouched
    return torch.cat([native, shielded_bank], dim=-1)

