"""Smart compression for attention-native memory banks.

The helpers here operate on plain bank ``state_dict`` dictionaries. They never
create parameters and never require gradients; callers can opt in from runtime
bank flags while the default bank path remains untouched.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F


def _bank_len(state: dict[str, Any]) -> int:
    m_k = state.get("M_K") or []
    return int(m_k[0].size(0)) if m_k else 0


def _row_features(state: dict[str, Any]) -> torch.Tensor:
    parts = []
    for k in state.get("M_K", []):
        if k.size(0) == 0:
            continue
        parts.append(k.detach().float().reshape(k.size(0), -1).cpu())
    if not parts:
        return torch.empty(0, 0)
    return F.normalize(torch.cat(parts, dim=1), dim=1, eps=1e-12)


def _as_counts(state: dict[str, Any], n: int) -> torch.Tensor:
    counts = state.get("merge_counts", state.get("merge_count"))
    if counts is None:
        return torch.ones(n, dtype=torch.float32)
    if isinstance(counts, torch.Tensor):
        out = counts.detach().float().cpu().flatten()
    else:
        out = torch.tensor(list(counts), dtype=torch.float32)
    if out.numel() != n:
        return torch.ones(n, dtype=torch.float32)
    return out.clamp_min(1.0)


def _merge_pair(state: dict[str, Any], i: int, j: int, counts: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
    out = deepcopy(state)
    n = _bank_len(state)
    keep = [idx for idx in range(n) if idx not in (i, j)]
    ci = counts[i]
    cj = counts[j]
    denom = (ci + cj).clamp_min(1.0)
    merged_count = ci + cj

    new_k = []
    new_v = []
    for k, v in zip(state.get("M_K", []), state.get("M_V", [])):
        device = k.device
        keep_t = torch.tensor(keep, dtype=torch.long, device=device)
        centroid_k = ((k[i].float() * ci + k[j].float() * cj) / denom).to(k.dtype)
        centroid_v = ((v[i].float() * ci + v[j].float() * cj) / denom).to(v.dtype)
        new_k.append(torch.cat([k.index_select(0, keep_t), centroid_k.unsqueeze(0)], dim=0))
        new_v.append(torch.cat([v.index_select(0, keep_t), centroid_v.unsqueeze(0)], dim=0))
    out["M_K"] = new_k
    out["M_V"] = new_v

    new_counts = torch.cat([counts[keep], merged_count.reshape(1)])
    out["merge_counts"] = new_counts

    for key in ("fact_ids", "address_strs"):
        vals = list(state.get(key, []))
        if len(vals) == n:
            merged = f"merge({vals[i]},{vals[j]})"
            out[key] = [vals[idx] for idx in keep] + [merged]

    for key in ("write_step", "last_access_step", "_x7_write_step", "_x7_last_access", "_x7_access_count"):
        vals = state.get(key)
        if vals is None or len(vals) != n:
            continue
        if key.endswith("access_count"):
            merged_val = vals[i] + vals[j]
        elif "last_access" in key:
            merged_val = max(vals[i], vals[j])
        else:
            merged_val = min(vals[i], vals[j])
        out[key] = [vals[idx] for idx in keep] + [merged_val]

    scores = state.get("importance_scores")
    if scores is not None and len(scores) == n:
        s = torch.as_tensor(scores, dtype=torch.float32).flatten()
        out["importance_scores"] = torch.cat([s[keep], torch.maximum(s[i], s[j]).reshape(1)])
    return out, new_counts


def compress_bank(bank_state: dict[str, Any], target_size: int) -> dict[str, Any]:
    """Return a compressed copy of ``bank_state`` using cosine K clustering.

    Rows are greedily agglomerated by highest cosine similarity until
    ``target_size`` is reached or no pair exceeds ``compression_min_similarity``
    (default 0.90). K centroids and V rows are weighted by ``merge_counts``.
    """
    with torch.no_grad():
        state = deepcopy(bank_state)
        n = _bank_len(state)
        target = int(target_size)
        if target <= 0 or n <= target:
            if n and "merge_counts" not in state:
                state["merge_counts"] = torch.ones(n, dtype=torch.float32)
            return state

        min_sim = float(state.get("compression_min_similarity", 0.90))
        counts = _as_counts(state, n)
        while _bank_len(state) > target:
            feats = _row_features(state)
            n_now = feats.size(0)
            if n_now <= target or n_now < 2:
                break
            sim = feats @ feats.T
            sim.fill_diagonal_(-1.0)
            flat_idx = int(sim.argmax().item())
            best = float(sim.reshape(-1)[flat_idx].item())
            if best < min_sim:
                break
            i = flat_idx // n_now
            j = flat_idx % n_now
            if i > j:
                i, j = j, i
            state, counts = _merge_pair(state, i, j, counts)
        return state
