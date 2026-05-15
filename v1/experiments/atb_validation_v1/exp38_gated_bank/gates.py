"""Exp38 — gated bank runtime + gate variants.

Core abstraction: GatedDownProj wraps model.model.layers[L].mlp.down_proj
with input-conditional gating:

    y = W_base @ x + sum_i g_i(x) * (a_i^T x) * b_i

where g_i: R^{d_inter} -> [0,1].

All variants (G0..G5) share this runtime; only the gate function differs.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BankTensors:
    B: torch.Tensor  # (d_out, N)  stacked b_i
    A: torch.Tensor  # (d_in,  N)  stacked a_i (key)
    fact_ids: list


def stack_bank(entries: dict, device, dtype) -> BankTensors:
    ids = list(entries.keys())
    b_list = [entries[i]["b"].to(device=device, dtype=dtype) for i in ids]
    a_list = [entries[i]["a"].to(device=device, dtype=dtype) for i in ids]
    B = torch.stack(b_list, dim=1)  # (d_out, N)
    A = torch.stack(a_list, dim=1)  # (d_in,  N)
    return BankTensors(B=B, A=A, fact_ids=ids)


# ---- gate functions: (scores [T,N], hidden [T,d_in], ctx) -> gate [T,N] in [0,1] ----

def gate_G0_baseline(scores, hidden, ctx):
    """All-on (Exp35b)."""
    return torch.ones_like(scores)


def gate_G1_threshold(scores, hidden, ctx):
    """Per-fact threshold theta_i."""
    theta = ctx["theta"]  # (N,)
    return (scores > theta.to(scores.device, scores.dtype)).to(scores.dtype)


def gate_G2_topk_retrieval(scores, hidden, ctx):
    """Top-k_r by score per-token."""
    k_r = ctx["k_r"]
    T, N = scores.shape
    if k_r >= N:
        return torch.ones_like(scores)
    topk_idx = scores.topk(k_r, dim=-1).indices  # (T, k_r)
    g = torch.zeros_like(scores)
    g.scatter_(-1, topk_idx, 1.0)
    return g


def gate_G3_learned(scores, hidden, ctx):
    """Learned per-fact head: g_i = sigmoid(w_i^T hidden + b_i)."""
    W = ctx["W_g"]  # (d_in, N)
    b = ctx["b_g"]  # (N,)
    logits = hidden.to(W.dtype) @ W + b
    return torch.sigmoid(logits).to(scores.dtype)


def gate_G5_mixture(scores, hidden, ctx):
    """G2 (top-k retrieval) × G3 (confidence weighting)."""
    g_topk = gate_G2_topk_retrieval(scores, hidden, ctx)
    g_conf = gate_G3_learned(scores, hidden, ctx)
    return g_topk * g_conf


GATE_FNS = {
    "G0": gate_G0_baseline,
    "G1": gate_G1_threshold,
    "G2": gate_G2_topk_retrieval,
    "G3": gate_G3_learned,
    "G4": gate_G3_learned,   # same fn, different training data
    "G5": gate_G5_mixture,
}


# ---- runtime: monkey-patch the down_proj forward with gating ----

class _GatedDownProjState:
    """Holds state shared between hook closure and outer scope."""
    def __init__(self, base_linear, bank: BankTensors, gate_fn, gate_ctx,
                 capture_occupancy: bool = False):
        self.base = base_linear
        self.bank = bank
        self.gate_fn = gate_fn
        self.gate_ctx = gate_ctx
        self.capture_occupancy = capture_occupancy
        self.occupancies = []
        self.n_calls = 0


def _make_gated_forward(state: _GatedDownProjState):
    base_lin = state.base
    bank = state.bank

    def forward(x):
        # x: (..., d_in)
        y_base = F.linear(x, base_lin.weight, base_lin.bias)
        if bank is None:
            return y_base

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])  # (T, d_in)
        scores = x2 @ bank.A  # (T, N)
        gates = state.gate_fn(scores, x2, state.gate_ctx)  # (T, N) in [0,1]
        if state.capture_occupancy:
            state.occupancies.append(gates.mean().item())
            state.n_calls += 1
        weighted = gates * scores  # (T, N)
        delta = weighted @ bank.B.t()  # (T, d_out)
        delta = delta.reshape(*orig_shape[:-1], -1).to(y_base.dtype)
        return y_base + delta

    return forward


@contextmanager
def gated_patches(model, edit_layer: int, bank: Optional[BankTensors],
                  gate_fn, gate_ctx: dict, capture_occupancy: bool = False):
    """Replace down_proj.forward with a gated variant; restore on exit.

    If bank is None, behaves like baseline (no patch).
    """
    layer = model.model.layers[edit_layer]
    mlp = layer.mlp
    base_lin = mlp.down_proj
    saved_forward = base_lin.forward

    if bank is None:
        try:
            yield None
        finally:
            base_lin.forward = saved_forward
        return

    state = _GatedDownProjState(base_lin, bank, gate_fn, gate_ctx, capture_occupancy)
    base_lin.forward = _make_gated_forward(state)
    try:
        yield state
    finally:
        base_lin.forward = saved_forward


@torch.no_grad()
def margin_at_last_gated(model, tokenizer, prompt, t_new, t_true):
    """Run a forward pass and return (logp_new - logp_true) at last position."""
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**enc, use_cache=False)
    last = int(enc["attention_mask"][0].sum().item() - 1)
    logp = F.log_softmax(out.logits[0, last].float(), dim=-1)
    return float(logp[t_new].item() - logp[t_true].item())
