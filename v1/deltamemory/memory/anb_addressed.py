"""Exp14+15+18 — oracle / natural slot restriction over an AttnNativeBank.

Building blocks:
  * ``subbank_select(bank, indices)``   → new bank holding only those slots.
  * ``subbank_correct(bank, fact_id)``  → one-slot bank with the correct fact.
  * ``subbank_random(bank, rng, k=1, exclude=())``
  * ``swap_K(bank_a, bank_b)``          → K from a, V from b (per layer).
  * ``shuffle_layer(bank, rng)``        → permute layer index of M_K, M_V.
  * ``shuffle_head(bank, rng)``         → permute head dim of M_K, M_V.

All operations return *new* AttnNativeBank objects (deep-copied tensors) so
the source bank is never mutated.  Compatible with ``forward_with_bank``
unchanged.
"""
from __future__ import annotations

import copy
from typing import Iterable, Sequence

import torch

from deltamemory.memory.attn_native_bank import AttnNativeBank


def _empty_like(bank: AttnNativeBank) -> AttnNativeBank:
    """Return a fresh bank with the same shape config but no slots."""
    return AttnNativeBank(
        num_layers=bank.num_layers,
        num_kv_heads=bank.num_kv_heads,
        head_dim=bank.head_dim,
        head_dims=list(bank.head_dims),
        num_kv_heads_per_layer=list(bank.num_kv_heads_per_layer),
        device=bank.device,
        dtype=bank.dtype,
        # Inherit ablation flags so downstream readout uses the same code path.
        value_scale_mode=bank.value_scale_mode,
        value_target_rms=bank.value_target_rms,
        bank_key_mode=bank.bank_key_mode,
    )


def subbank_select(bank: AttnNativeBank, indices: Sequence[int]) -> AttnNativeBank:
    """Return a new bank containing only the given slot indices (preserving order)."""
    out = _empty_like(bank)
    idx = list(indices)
    if not idx:
        return out  # empty
    sel = torch.tensor(idx, dtype=torch.long)
    for layer in range(bank.num_layers):
        sel_layer = sel.to(bank.M_K[layer].device)
        out.M_K[layer] = bank.M_K[layer].index_select(0, sel_layer).clone()
        out.M_V[layer] = bank.M_V[layer].index_select(0, sel_layer).clone()
    out.fact_ids = [bank.fact_ids[i] for i in idx]
    out.address_strs = [bank.address_strs[i] for i in idx if i < len(bank.address_strs)]
    return out


def subbank_correct(bank: AttnNativeBank, fact_id: str) -> AttnNativeBank:
    """Single-slot oracle bank with only ``fact_id``."""
    if fact_id not in bank.fact_ids:
        raise KeyError(f"fact_id {fact_id!r} not in bank")
    i = bank.fact_ids.index(fact_id)
    return subbank_select(bank, [i])


def subbank_random(
    bank: AttnNativeBank,
    rng: torch.Generator,
    *,
    k: int = 1,
    exclude: Iterable[str] = (),
) -> AttnNativeBank:
    """Random ``k``-slot bank, excluding any fact_id in ``exclude``."""
    excl = set(exclude)
    eligible = [i for i, f in enumerate(bank.fact_ids) if f not in excl]
    if not eligible:
        raise ValueError("no eligible facts left after exclusion")
    k = min(k, len(eligible))
    perm = torch.randperm(len(eligible), generator=rng).tolist()[:k]
    chosen = [eligible[p] for p in perm]
    return subbank_select(bank, chosen)


def subbank_swap_KV(
    bank_K_source: AttnNativeBank,
    bank_V_source: AttnNativeBank,
) -> AttnNativeBank:
    """Return a bank with K from ``bank_K_source`` and V from ``bank_V_source``.

    Both source banks must have the same per-layer shapes and slot counts.
    Slot identities follow ``bank_K_source.fact_ids`` (this is the "addressing"
    side).  Used to test the ``correct_K + random_V`` and ``random_K +
    correct_V`` variants in Exp14.
    """
    if bank_K_source.num_layers != bank_V_source.num_layers:
        raise ValueError("layer count mismatch")
    if len(bank_K_source.fact_ids) != len(bank_V_source.fact_ids):
        raise ValueError("slot count mismatch")
    out = _empty_like(bank_K_source)
    for layer in range(bank_K_source.num_layers):
        out.M_K[layer] = bank_K_source.M_K[layer].clone()
        out.M_V[layer] = bank_V_source.M_V[layer].clone()
    out.fact_ids = list(bank_K_source.fact_ids)
    out.address_strs = list(bank_K_source.address_strs)
    return out


def subbank_shuffle_layer(bank: AttnNativeBank, rng: torch.Generator) -> AttnNativeBank:
    """Permute layer index of M_K and M_V (same permutation for K and V).

    Only layers with matching shape get permuted together; otherwise the
    layer is left in place.  Norms are preserved per layer (we never touch
    per-tensor magnitudes).
    """
    out = _empty_like(bank)
    out.fact_ids = list(bank.fact_ids)
    out.address_strs = list(bank.address_strs)

    # Group layers by shape signature so permutation stays consistent.
    sig = [
        (bank.M_K[l].shape, bank.M_V[l].shape) for l in range(bank.num_layers)
    ]
    by_sig: dict[tuple, list[int]] = {}
    for l, s in enumerate(sig):
        by_sig.setdefault(s, []).append(l)
    perm_map = list(range(bank.num_layers))
    for layers in by_sig.values():
        if len(layers) <= 1:
            continue
        p = torch.randperm(len(layers), generator=rng).tolist()
        permuted = [layers[i] for i in p]
        for src, dst in zip(layers, permuted):
            perm_map[src] = dst
    for layer in range(bank.num_layers):
        src = perm_map[layer]
        out.M_K[layer] = bank.M_K[src].clone()
        out.M_V[layer] = bank.M_V[src].clone()
    return out


def subbank_shuffle_head(bank: AttnNativeBank, rng: torch.Generator) -> AttnNativeBank:
    """Permute head dim of M_K and M_V independently per layer.

    Same permutation applied to K and V within a layer so the K↔V binding
    survives, but head identity is destroyed.  Norms unchanged.
    """
    out = _empty_like(bank)
    out.fact_ids = list(bank.fact_ids)
    out.address_strs = list(bank.address_strs)
    for layer in range(bank.num_layers):
        mk = bank.M_K[layer]
        mv = bank.M_V[layer]
        if mk.numel() == 0:
            out.M_K[layer] = mk.clone()
            out.M_V[layer] = mv.clone()
            continue
        h = mk.size(1)
        if h <= 1:
            out.M_K[layer] = mk.clone()
            out.M_V[layer] = mv.clone()
            continue
        p = torch.randperm(h, generator=rng)
        p = p.to(mk.device)
        out.M_K[layer] = mk.index_select(1, p).clone()
        out.M_V[layer] = mv.index_select(1, p).clone()
    return out


def subbank_shuffle_V(bank: AttnNativeBank, rng: torch.Generator) -> AttnNativeBank:
    """Keep K, but permute V across the slot dimension (per layer).

    Tests Effect(correct_K, shuffled_V): if matched K/V binding is required,
    this should kill the readout.
    """
    out = _empty_like(bank)
    out.fact_ids = list(bank.fact_ids)
    out.address_strs = list(bank.address_strs)
    n = len(bank.fact_ids)
    if n == 0:
        for layer in range(bank.num_layers):
            out.M_K[layer] = bank.M_K[layer].clone()
            out.M_V[layer] = bank.M_V[layer].clone()
        return out
    perm = torch.randperm(n, generator=rng)
    for layer in range(bank.num_layers):
        out.M_K[layer] = bank.M_K[layer].clone()
        out.M_V[layer] = bank.M_V[layer].index_select(
            0, perm.to(bank.M_V[layer].device)
        ).clone()
    return out


def subbank_mask_layers(bank: AttnNativeBank, keep_layers: Iterable[int]) -> AttnNativeBank:
    """Zero out M_K and M_V on layers NOT in ``keep_layers``.

    Slots remain present so attention math is identical except for the masked
    layers contributing zero to the bank-side softmax (their scores become
    0 against any Q, which after exp() are uniform within the bank slice and
    therefore add no fact-specific signal — close to but not exactly bit-equal
    to dropping them).  Used by Exp16 site map.
    """
    keep = set(int(l) for l in keep_layers)
    out = _empty_like(bank)
    out.fact_ids = list(bank.fact_ids)
    out.address_strs = list(bank.address_strs)
    for layer in range(bank.num_layers):
        mk = bank.M_K[layer].clone()
        mv = bank.M_V[layer].clone()
        if layer not in keep:
            mk = torch.zeros_like(mk)
            mv = torch.zeros_like(mv)
        out.M_K[layer] = mk
        out.M_V[layer] = mv
    return out


def subbank_mask_heads(
    bank: AttnNativeBank, layer_head_keep: dict[int, Iterable[int]]
) -> AttnNativeBank:
    """Zero out heads not listed in ``layer_head_keep[layer]``; pass others through.

    Layers not present in the dict have ALL heads kept.  Layers with an empty
    list have all heads zeroed.
    """
    out = _empty_like(bank)
    out.fact_ids = list(bank.fact_ids)
    out.address_strs = list(bank.address_strs)
    for layer in range(bank.num_layers):
        mk = bank.M_K[layer].clone()
        mv = bank.M_V[layer].clone()
        if layer in layer_head_keep:
            keep = set(int(h) for h in layer_head_keep[layer])
            if mk.numel() > 0 and mk.dim() == 3:
                for h in range(mk.size(1)):
                    if h not in keep:
                        mk[:, h, :] = 0.0
                        mv[:, h, :] = 0.0
        out.M_K[layer] = mk
        out.M_V[layer] = mv
    return out


__all__ = [
    "subbank_select", "subbank_correct", "subbank_random",
    "subbank_swap_KV", "subbank_shuffle_layer", "subbank_shuffle_head",
    "subbank_shuffle_V", "subbank_mask_layers", "subbank_mask_heads",
]
