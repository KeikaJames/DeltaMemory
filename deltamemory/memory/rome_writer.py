"""Stage 14E — ROME-style closed-form writer rebuild.

13C's pure-nullify writer produced held-out recall@1 = 0.184. The cause was
twofold: (1) the writer subspace was rank-4 (too few directions), and (2)
the writer only nullified the address direction without rebuilding the
target value.

Stage 14E follows the ROME identity. Given pairs of (address keys K, target
values V) collected from the train split:

    K  ∈ ℝ^{N × d}   stack of post-norm pre-RoPE bank K's at the address site
    V  ∈ ℝ^{N × d}   stack of post-norm bank V's at the value site

we solve, per attention layer, a single ridge regression:

    W_v_delta = (Kᵀ K + λ I)^{-1} Kᵀ V       ∈ ℝ^{d × d}

The bank-V slot for fact i is then rewritten as:

    M_V_new[i] = K[i] @ W_v_delta            (per-head)

This is the same closed-form ROME uses for MLP-row weight edits, but applied
to the **non-destructive bank slot** instead of the model's MLP weights.
The base model parameters are not modified.

Invariants
----------
* α = 0 / empty-bank bit-equality is preserved trivially: the rewrite
  changes ``bank.M_V`` content but leaves the forward path unchanged. With
  α = 0 the bank V is never consumed.
* Per-head: the regression is solved independently per ``num_kv_heads``
  head so heads do not bleed into one another.
* Numerical stability: solved in float32 even when bank tensors are bf16.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RomeWriterStats:
    """Diagnostics returned by :func:`solve_rome_writer`."""

    layer_idx: int
    head_idx: int
    n_facts: int
    cond_number: float
    residual_rms: float
    lambda_reg: float


def solve_rome_writer(
    K: torch.Tensor,
    V: torch.Tensor,
    lambda_reg: float = 1e-2,
) -> tuple[torch.Tensor, RomeWriterStats]:
    """Solve W = (Kᵀ K + λI)^{-1} Kᵀ V for one (layer, head) pair.

    Args:
        K: address keys, shape ``[N, d]``.
        V: target values, shape ``[N, d]``.
        lambda_reg: ridge coefficient (must be > 0).

    Returns:
        ``(W, stats)`` where ``W`` is shape ``[d, d]`` in K's dtype, and
        ``stats`` carries condition number + residual RMS for logging.
    """
    if K.shape != V.shape:
        raise ValueError(f"K and V must have matching shape; got {K.shape} vs {V.shape}")
    if K.ndim != 2:
        raise ValueError(f"K must be 2-D [N, d]; got {K.shape}")
    if lambda_reg <= 0:
        raise ValueError(f"lambda_reg must be positive; got {lambda_reg}")

    orig_dtype = K.dtype
    Kf = K.to(torch.float32)
    Vf = V.to(torch.float32)
    n, d = Kf.shape

    gram = Kf.t() @ Kf + lambda_reg * torch.eye(d, device=Kf.device, dtype=torch.float32)
    rhs = Kf.t() @ Vf
    W = torch.linalg.solve(gram, rhs)             # [d, d]

    pred = Kf @ W
    residual = (pred - Vf)
    residual_rms = float(residual.pow(2).mean().sqrt().item())
    try:
        cond = float(torch.linalg.cond(gram).item())
    except Exception:
        cond = float("nan")

    stats = RomeWriterStats(
        layer_idx=-1,
        head_idx=-1,
        n_facts=n,
        cond_number=cond,
        residual_rms=residual_rms,
        lambda_reg=lambda_reg,
    )
    return W.to(orig_dtype), stats


def rebuild_bank_v(
    bank,
    *,
    lambda_reg: float = 1e-2,
) -> list[list[RomeWriterStats]]:
    """Rebuild ``bank.M_V`` in place from (K, V) using the ROME identity.

    For each (layer, head): solve ``W = (KᵀK + λI)⁻¹ KᵀV`` and replace
    ``bank.M_V[layer][:, head, :]`` with ``K @ W``.

    Args:
        bank: an :class:`AttnNativeBank` with ``M_K`` and ``M_V`` already
            populated. Must contain at least one fact.
        lambda_reg: ridge coefficient.

    Returns:
        ``stats[layer][head]`` for diagnostics. Empty layers (e.g.
        KV-shared layers with no slots of their own) yield empty lists.
    """
    all_stats: list[list[RomeWriterStats]] = []
    for layer in range(bank.num_layers):
        K_layer = bank.M_K[layer]   # [N, Hkv, d]
        V_layer = bank.M_V[layer]
        layer_stats: list[RomeWriterStats] = []
        if K_layer.numel() == 0:
            all_stats.append(layer_stats)
            continue
        n, hkv, d = K_layer.shape
        new_V = V_layer.clone()
        for h in range(hkv):
            Kh = K_layer[:, h, :]
            Vh = V_layer[:, h, :]
            W, stats = solve_rome_writer(Kh, Vh, lambda_reg=lambda_reg)
            stats.layer_idx = layer
            stats.head_idx = h
            new_V[:, h, :] = (Kh.to(torch.float32) @ W.to(torch.float32)).to(V_layer.dtype)
            layer_stats.append(stats)
        bank.M_V[layer] = new_V
        all_stats.append(layer_stats)
    return all_stats
