"""Exp25 helpers: SlotRecorder + new subbank variants.

The injector publishes per-layer ``record_bank_attn(layer_idx, weights,
T_orig)`` callbacks via ``deltamemory.diagnostics._RECORDER``. We expose a
minimal recorder that captures the LAST-token bank attention slice and
returns per-eval aggregates:

  selected_slot_id        mode over layers of argmax bank slot at last token
  per_layer_selected      list of int (one per layer)
  bank_attention_mass     mean sum(bank weights) at last token, over layers
  max_bank_prob           mean max bank weight at last token, over layers

Usage:

    with SlotRecorder(model) as rec:
        # run any forward(s) using injector
        ...
    metrics = rec.aggregate()

The recorder also collects per-layer top-2 to support a top1-top2 score gap
proxy (computed on softmaxed weights, not raw scores).
"""
from __future__ import annotations

from typing import Optional
import statistics
import torch

import deltamemory.diagnostics as _diag_mod
from deltamemory.memory.anb_addressed import _empty_like


class SlotRecorder:
    """Light-weight recorder for retrieval-accuracy metrics.

    Only records the LAST-token slice of bank weights at each layer of each
    forward pass. Drops everything else.
    """

    def __init__(self) -> None:
        self._records: list[dict] = []
        self._prev = None
        self._n_forwards: int = 0

    def __enter__(self) -> "SlotRecorder":
        self._prev = _diag_mod._RECORDER
        _diag_mod._RECORDER = self
        return self

    def __exit__(self, *_):
        _diag_mod._RECORDER = self._prev
        self._prev = None

    # API expected by attn_native_bank.py ---------------------------------

    def record_bank_attn(self, layer_idx: int, weights: torch.Tensor, T_orig: int) -> None:
        # weights: (B, H, T, T_orig + N). Take last token's bank slice.
        w = weights.detach().float()
        w_bank_last = w[..., -1, T_orig:]  # (B, H, N)
        if w_bank_last.numel() == 0:
            return
        # Mean over (B, H) → (N,).
        w_mean = w_bank_last.mean(dim=(0, 1))
        N = int(w_mean.size(0))
        if N == 0:
            return
        top1_w, top1_i = torch.topk(w_mean, k=min(1, N))
        if N >= 2:
            top2_w = torch.topk(w_mean, k=2).values[1].item()
        else:
            top2_w = 0.0
        self._records.append({
            "layer": int(layer_idx),
            "selected_slot": int(top1_i.item()),
            "top1_weight": float(top1_w.item()),
            "top2_weight": float(top2_w),
            "bank_mass": float(w_bank_last.sum(dim=-1).mean().item()),
        })

    def record_bank_readout(self, *args, **kwargs) -> None:  # noqa: D401
        # Unused — required by injector API.
        return

    # Aggregation ---------------------------------------------------------

    def aggregate(self) -> dict:
        if not self._records:
            return {
                "selected_slot_id": -1,
                "per_layer_selected": [],
                "bank_attention_mass": 0.0,
                "max_bank_prob": 0.0,
                "top1_top2_gap": 0.0,
                "n_layer_records": 0,
            }
        per_layer = [r["selected_slot"] for r in self._records]
        try:
            mode_slot = statistics.mode(per_layer)
        except statistics.StatisticsError:
            mode_slot = per_layer[-1]
        masses = [r["bank_mass"] for r in self._records]
        top1s = [r["top1_weight"] for r in self._records]
        top2s = [r["top2_weight"] for r in self._records]
        return {
            "selected_slot_id": int(mode_slot),
            "per_layer_selected": per_layer,
            "bank_attention_mass": float(sum(masses) / len(masses)),
            "max_bank_prob": float(sum(top1s) / len(top1s)),
            "top1_top2_gap": float(
                sum(t1 - t2 for t1, t2 in zip(top1s, top2s)) / len(top1s)
            ),
            "n_layer_records": len(self._records),
        }


def subbank_meanV(bank):
    """Return a bank with M_K unchanged but each slot's M_V replaced by the
    cross-slot mean M_V (per layer, per head).

    Used to decompose "K-routing increment" from "generic V steering".
    If retrieval signal disappears here, the K-routing-conditional V
    readout was the source. If signal stays, ANB is purely a V-bias library.
    """
    out = _empty_like(bank)
    for layer in range(bank.num_layers):
        K = bank.M_K[layer]
        V = bank.M_V[layer]
        if V.numel() == 0:
            out.M_K[layer] = K.clone()
            out.M_V[layer] = V.clone()
            continue
        V_mean = V.mean(dim=0, keepdim=True).expand_as(V).contiguous()  # (N, H, D)
        out.M_K[layer] = K.clone()
        out.M_V[layer] = V_mean.clone()
    out.fact_ids = list(bank.fact_ids)
    out.address_strs = list(bank.address_strs)
    return out


def subbank_select_one_random(bank, rng: torch.Generator, exclude_fact_id: Optional[str] = None):
    """Pick one *random* slot from the bank (keeping the full bank size = 1)."""
    n = len(bank.fact_ids)
    if n == 0:
        return bank
    eligible = [i for i, f in enumerate(bank.fact_ids) if f != exclude_fact_id]
    if not eligible:
        return bank
    idx = int(torch.randint(0, len(eligible), (1,), generator=rng).item())
    return bank.__class__  # placeholder; not used currently
