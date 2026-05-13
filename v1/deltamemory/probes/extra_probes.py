"""Extra L-marathon probes: top_k_softmax_entropy and bank_eldest_score_margin.

These are deliberately decoupled from the AttnNativeBank forward path. They
operate on tensors / a Bank object handed in by the caller, so the probes can
evolve without coupling to the hot path.
"""

from __future__ import annotations

from typing import Any

import torch


def top_k_softmax_entropy(
    attn_weights: torch.Tensor,
    k: int = 8,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Entropy across the top-k attention weights only (in nats)."""
    if attn_weights.numel() == 0:
        return torch.zeros(attn_weights.shape[:-1], dtype=attn_weights.dtype)
    n = attn_weights.shape[-1]
    k_eff = max(1, min(int(k), n))
    topk_vals, _ = torch.topk(attn_weights, k_eff, dim=-1)
    s = topk_vals.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = topk_vals / s
    h = -(p.clamp_min(eps).log() * p).sum(dim=-1)
    return h


def bank_eldest_score_margin(
    bank: Any,
    forward_with_bank,
    tokenizer,
    prompt: str,
    target_canonical: str,
    target_eldest: str,
    device: str = "cpu",
) -> float:
    """Log-margin of the *eldest* bank entry's recall vs canonical answer."""
    ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    can_ids = tokenizer.encode(target_canonical, add_special_tokens=False)
    elder_ids = tokenizer.encode(target_eldest, add_special_tokens=False)
    if not can_ids or not elder_ids:
        return float("nan")
    with torch.no_grad():
        logits = forward_with_bank(ids, bank)
    last = logits[0, -1].float()
    logp = torch.log_softmax(last, dim=-1)
    return float(logp[elder_ids[0]].item() - logp[can_ids[0]].item())
