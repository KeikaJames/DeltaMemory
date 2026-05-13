"""Exp26b multi-token V capture extension to dual-site API.

Captures K at single position (pos_K), V averaged across multiple
positions (pos_V_list). One slot per fact. Architecture stays native.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from deltamemory.memory.attn_native_bank import (
    AttnNativeBank,
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)


def write_fact_dual_site_multi_v(
    patcher: AttnNativePatcher,
    bank: AttnNativeBank,
    tokenizer,
    write_prompt: str,
    fact_id: str,
    address: Optional[str],
    capture_pos_K: int,
    capture_pos_V_list: Sequence[int],
) -> None:
    """K@pos_K (single), V averaged over pos_V_list (multi).

    Returns one slot. If pos_V_list has length 1, equivalent to
    ``write_fact_dual_site``.
    """
    pos_V_list = [int(p) for p in capture_pos_V_list if p is not None]
    if not pos_V_list:
        raise ValueError("pos_V_list must be non-empty")

    # K pass
    scratch_K = fresh_bank(patcher.model)
    scratch_K.value_scale_mode = bank.value_scale_mode
    scratch_K.value_target_rms = bank.value_target_rms
    scratch_K.bank_key_mode = bank.bank_key_mode
    write_fact(patcher, scratch_K, tokenizer,
               write_prompt=write_prompt, fact_id=fact_id,
               address=address, capture_pos=int(capture_pos_K))
    K_per_layer = [scratch_K.M_K[layer][-1].clone() for layer in range(scratch_K.num_layers)]

    # V passes: one write per V position, then mean across passes per layer.
    V_accum = None
    for pV in pos_V_list:
        scratch_V = fresh_bank(patcher.model)
        scratch_V.value_scale_mode = bank.value_scale_mode
        scratch_V.value_target_rms = bank.value_target_rms
        scratch_V.bank_key_mode = bank.bank_key_mode
        write_fact(patcher, scratch_V, tokenizer,
                   write_prompt=write_prompt, fact_id=fact_id,
                   address=address, capture_pos=int(pV))
        V_layers = [scratch_V.M_V[layer][-1].clone() for layer in range(scratch_V.num_layers)]
        if V_accum is None:
            V_accum = V_layers
        else:
            V_accum = [a + b for a, b in zip(V_accum, V_layers)]
    n = len(pos_V_list)
    V_per_layer = [v / n for v in V_accum]

    patcher._apply_smart_flags(bank)
    bank.append(K_per_layer, V_per_layer, fact_id=fact_id, address=address)
