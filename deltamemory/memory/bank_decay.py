"""Ebbinghaus-style decay for attention-native memory banks."""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch


def _bank_len(state: dict[str, Any]) -> int:
    m_v = state.get("M_V") or []
    return int(m_v[0].size(0)) if m_v else 0


def _steps(state: dict[str, Any], key: str, fallback: str, n: int) -> torch.Tensor:
    vals = state.get(key, state.get(fallback))
    if vals is None or len(vals) != n:
        return torch.zeros(n, dtype=torch.float32)
    return torch.as_tensor(vals, dtype=torch.float32).flatten()


def _original_norms(state: dict[str, Any], n: int) -> torch.Tensor:
    vals = state.get("original_v_norm")
    if vals is not None and len(vals) == n:
        return torch.as_tensor(vals, dtype=torch.float32).flatten().clamp_min(1e-12)
    norms = torch.zeros(n, dtype=torch.float32)
    for v in state.get("M_V", []):
        norms += torch.linalg.vector_norm(v.detach().float().reshape(n, -1), ord=2, dim=1).cpu()
    return norms.clamp_min(1e-12)


def _filter_rows(state: dict[str, Any], keep: torch.Tensor) -> dict[str, Any]:
    keep_list = keep.cpu().tolist()
    out = deepcopy(state)
    out["M_K"] = [k.index_select(0, keep.to(k.device)) for k in state.get("M_K", [])]
    out["M_V"] = [v.index_select(0, keep.to(v.device)) for v in state.get("M_V", [])]
    old_n = _bank_len(state)
    # Per-layer metadata: length equals num_layers, must NOT be filtered as rows
    # even if num_layers == old_n (n_facts).
    LAYER_META_KEYS = {"head_dims", "num_kv_heads_per_layer"}
    for key, vals in list(state.items()):
        if key in {"M_K", "M_V"} or key in LAYER_META_KEYS:
            continue
        if isinstance(vals, torch.Tensor) and vals.ndim >= 1 and vals.size(0) == old_n:
            out[key] = vals.index_select(0, keep.to(vals.device))
        elif isinstance(vals, list) and len(vals) == old_n:
            out[key] = [vals[i] for i in keep_list]
    if keep.numel() == 0:
        out.setdefault("fact_ids", [])
        out.setdefault("address_strs", [])
    return out


def apply_decay(bank_state: dict[str, Any], current_step: int, half_life: int = 1000) -> dict[str, Any]:
    """Return a bank copy with V softly decayed and obsolete rows erased.

    ``V_decayed = V * exp(-lambda * age / half_life)`` with
    ``lambda = ln(2)`` by default. Rows whose decayed/original norm ratio falls
    below ``decay_erase_threshold`` (default ``1e-3``) are removed.
    """
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    with torch.no_grad():
        state = deepcopy(bank_state)
        n = _bank_len(state)
        if n == 0:
            return state
        last_access = _steps(state, "last_access_step", "_x7_last_access", n)
        age = (float(current_step) - last_access).clamp_min(0.0)
        lam = float(state.get("decay_lambda", math.log(2.0)))
        factors = torch.exp(-lam * age / float(half_life)).clamp(min=0.0, max=1.0)
        state["M_V"] = [
            (v.float() * factors.to(v.device).reshape(-1, 1, 1)).to(v.dtype)
            for v in state.get("M_V", [])
        ]
        original = _original_norms(bank_state, n)
        decayed = torch.zeros(n, dtype=torch.float32)
        for v in state.get("M_V", []):
            decayed += torch.linalg.vector_norm(v.detach().float().reshape(n, -1), ord=2, dim=1).cpu()
        ratio = decayed / original
        threshold = float(state.get("decay_erase_threshold", 1e-3))
        keep = torch.nonzero(ratio >= threshold, as_tuple=False).flatten().to(torch.long)
        state["decay_factors"] = factors
        if keep.numel() < n:
            state = _filter_rows(state, keep)
            state["decay_factors"] = factors.index_select(0, keep)
        return state
