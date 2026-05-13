"""Exp13 — QK-only addressability diagnostics for ANB.

Pure measurement: forwards a read prompt with the patched-but-empty bank,
records per-layer Q at a chosen token position, then scores Q against a
populated bank's M_K offline.  V is never injected, logits are never
perturbed — this isolates the natural addressability signal:

    s_i(layer, head) = <q_pre(layer, head), M_K[layer, head, i]>

without coupling it to readout dynamics.

Used by ``experiments/atb_validation_v1/exp13_anb_readdressability/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch

from deltamemory.memory.attn_native_bank import AttnNativeBank, AttnNativePatcher


QueryMode = Literal["pre_rope", "post_rope"]


@dataclass
class LayerScores:
    """Bank scores for one read prompt at one layer.

    Attributes:
        layer: layer index.
        scores: tensor of shape ``[Hq, N]`` — raw inner-product scores between
            this layer's Q (one token position) and bank M_K (already
            head-repeated to query-head resolution).
        scores_mean_heads: tensor of shape ``[N]`` — head-averaged scores.
    """
    layer: int
    scores: torch.Tensor
    scores_mean_heads: torch.Tensor


@dataclass
class ReadProbe:
    """Recorded Q per layer for one read prompt at a single token pos."""
    prompt: str
    q_pre: list[torch.Tensor | None]   # each entry [Hq, d]
    q_post: list[torch.Tensor | None]  # each entry [Hq, d]


def record_read_queries(
    patcher: AttnNativePatcher,
    tokenizer,
    read_prompt: str,
    *,
    capture_pos: int | None = None,
) -> ReadProbe:
    """Forward ``read_prompt`` with the patched model, recording Q per layer.

    ``capture_pos = None`` → last real token (default last_token_Q).
    ``capture_pos = int`` → that absolute token index.

    Bank is left empty / inactive; this is a pure read pass.
    """
    device = next(patcher.model.parameters()).device
    enc = tokenizer(read_prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    last = int(am.sum(dim=1).item()) - 1
    pos = last if capture_pos is None else int(capture_pos)
    pos = max(0, min(pos, last))

    with patcher.patched(), patcher.recording_queries(capture_pos=pos), torch.no_grad():
        patcher.model(input_ids=ids, attention_mask=am, use_cache=False)
    q_pre = [t[0].clone().detach() if t is not None else None
             for t in patcher._recorded_Q_pre]
    q_post = [t[0].clone().detach() if t is not None else None
              for t in patcher._recorded_Q_post]
    return ReadProbe(prompt=read_prompt, q_pre=q_pre, q_post=q_post)


def _repeat_kv_per_head(mk: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Expand bank K from KV-head resolution to Q-head resolution.

    Input  ``mk`` : [N, Hkv, d]
    Output       : [N, Hq, d]  where Hq = Hkv * num_groups.
    """
    if num_groups == 1:
        return mk
    n, hkv, d = mk.shape
    out = mk.unsqueeze(2).expand(n, hkv, num_groups, d).reshape(n, hkv * num_groups, d)
    return out


def score_query_against_bank(
    probe: ReadProbe,
    bank: AttnNativeBank,
    patcher: AttnNativePatcher,
    *,
    mode: QueryMode = "pre_rope",
    use_cosine: bool = False,
) -> dict[int, LayerScores]:
    """Score recorded Q against bank.M_K across all populated layers.

    Returns ``{layer_idx: LayerScores}`` (only for non-empty layers).
    Heads are aligned by expanding M_K to Q-head resolution via repeat_kv.
    """
    out: dict[int, LayerScores] = {}
    q_list = probe.q_pre if mode == "pre_rope" else probe.q_post
    for layer in range(bank.num_layers):
        q = q_list[layer]
        if q is None:
            continue
        mk = bank.M_K[layer]
        if mk.numel() == 0 or mk.size(0) == 0:
            continue
        # mk : [N, Hkv, d]; q : [Hq, d]
        # Determine repeat factor:
        hq = q.size(0)
        hkv = mk.size(1)
        if hq % max(hkv, 1) != 0:
            # Layer / head shape mismatch (e.g. shared KV layers stored at
            # source-layer resolution).  Skip — diagnostic only.
            continue
        groups = hq // hkv
        mk_full = _repeat_kv_per_head(mk.to(q.dtype).to(q.device), groups)  # [N, Hq, d]
        if use_cosine:
            qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            kn = mk_full / mk_full.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            # qn [Hq,d], kn [N,Hq,d]  -> scores [Hq, N]
            scores = torch.einsum("hd,nhd->hn", qn, kn)
        else:
            scores = torch.einsum("hd,nhd->hn", q, mk_full)
        out[layer] = LayerScores(
            layer=layer,
            scores=scores.detach().cpu(),
            scores_mean_heads=scores.mean(dim=0).detach().cpu(),
        )
    return out


def rank_correct(
    layer_scores: dict[int, LayerScores],
    fact_ids: list[str],
    correct_fact_id: str,
    *,
    reduce: Literal["max_layer", "mean_layer", "max_head_max_layer"] = "max_layer",
) -> dict[str, float]:
    """Compute correct-fact rank statistics across layers/heads.

    Returns a dict with:
      - ``correct_rank``: 0-indexed rank of correct fact under chosen reduction
      - ``correct_score``: score of correct fact
      - ``best_other_score``: highest score over non-correct facts
      - ``score_gap`` = correct_score - best_other_score
      - ``top_layer``: layer achieving max score over correct fact
      - ``top_index``: index in bank achieving overall max
      - ``recall_at_1``: 1.0 if correct is argmax else 0.0
      - ``recall_at_5``: 1.0 if correct in top-5 else 0.0
    """
    if not layer_scores:
        return {"correct_rank": -1.0, "correct_score": float("nan"),
                "best_other_score": float("nan"), "score_gap": float("nan"),
                "top_layer": -1.0, "top_index": -1.0,
                "recall_at_1": 0.0, "recall_at_5": 0.0}

    # Per-fact score under chosen reduction.
    # First stack per-layer head-mean scores: [L_present, N]
    layers_sorted = sorted(layer_scores.keys())
    stack = torch.stack([layer_scores[l].scores_mean_heads for l in layers_sorted], dim=0)
    if reduce == "max_layer":
        per_fact = stack.max(dim=0)            # values, indices over layers
        fact_scores = per_fact.values
        top_layer_per_fact = per_fact.indices
    elif reduce == "mean_layer":
        fact_scores = stack.mean(dim=0)
        top_layer_per_fact = torch.zeros_like(fact_scores, dtype=torch.long)
    elif reduce == "max_head_max_layer":
        # Per-layer max over heads, then max over layers.
        per_layer_max_head = torch.stack(
            [layer_scores[l].scores.max(dim=0).values for l in layers_sorted], dim=0)
        per_fact = per_layer_max_head.max(dim=0)
        fact_scores = per_fact.values
        top_layer_per_fact = per_fact.indices
    else:
        raise ValueError(f"unknown reduce={reduce!r}")

    n = fact_scores.size(0)
    try:
        correct_idx = fact_ids.index(correct_fact_id)
    except ValueError:
        return {"correct_rank": -1.0, "correct_score": float("nan"),
                "best_other_score": float("nan"), "score_gap": float("nan"),
                "top_layer": -1.0, "top_index": -1.0,
                "recall_at_1": 0.0, "recall_at_5": 0.0}
    correct_score = float(fact_scores[correct_idx].item())
    # Mask correct out and find best of the rest.
    mask = torch.ones(n, dtype=torch.bool)
    mask[correct_idx] = False
    if mask.any():
        best_other = float(fact_scores[mask].max().item())
    else:
        best_other = float("-inf")
    # Rank = number of facts strictly greater than correct.
    rank = int((fact_scores > correct_score).sum().item())
    top_index = int(fact_scores.argmax().item())
    top_layer = int(layers_sorted[int(top_layer_per_fact[correct_idx].item())])
    recall_at_1 = 1.0 if top_index == correct_idx else 0.0
    # top-5
    topk = min(5, n)
    top_idx_set = set(fact_scores.topk(topk).indices.tolist())
    recall_at_5 = 1.0 if correct_idx in top_idx_set else 0.0
    return {
        "correct_rank": float(rank),
        "correct_score": correct_score,
        "best_other_score": best_other,
        "score_gap": correct_score - best_other,
        "top_layer": float(top_layer),
        "top_index": float(top_index),
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
    }


def hard_negative_win_rate(
    layer_scores: dict[int, LayerScores],
    fact_ids: list[str],
    correct_fact_id: str,
    hard_negative_ids: Iterable[str],
    *,
    reduce: Literal["max_layer", "mean_layer"] = "max_layer",
) -> float:
    """Fraction of hard-negatives that out-score the correct fact.

    Higher is worse for the addressability hypothesis.
    """
    layers_sorted = sorted(layer_scores.keys())
    if not layers_sorted:
        return float("nan")
    stack = torch.stack([layer_scores[l].scores_mean_heads for l in layers_sorted], dim=0)
    fact_scores = stack.max(dim=0).values if reduce == "max_layer" else stack.mean(dim=0)
    try:
        correct_idx = fact_ids.index(correct_fact_id)
    except ValueError:
        return float("nan")
    correct_score = float(fact_scores[correct_idx].item())
    hn_ids = [h for h in hard_negative_ids if h in fact_ids and h != correct_fact_id]
    if not hn_ids:
        return float("nan")
    wins = 0
    for h in hn_ids:
        i = fact_ids.index(h)
        if float(fact_scores[i].item()) > correct_score:
            wins += 1
    return wins / len(hn_ids)


__all__ = [
    "ReadProbe", "LayerScores",
    "record_read_queries", "score_query_against_bank",
    "rank_correct", "hard_negative_win_rate",
]
