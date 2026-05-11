"""Exp23+ dual-site capture API.

Each fact slot stores ``M_K`` from ``site_K`` and ``M_V`` from ``site_V``,
allowing routing (K) and readout (V) to live at different attention sites
— the structural fix for the Exp18 NATURAL_FAIL.

The trick is implemented with two calls to ``write_fact`` into a scratch
bank, followed by splicing K from pass-1 and V from pass-2 into a single
slot on the target bank.
"""
from __future__ import annotations

from typing import Optional

from deltamemory.memory.attn_native_bank import (
    AttnNativeBank,
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)


def write_fact_dual_site(
    patcher: AttnNativePatcher,
    bank: AttnNativeBank,
    tokenizer,
    write_prompt: str,
    fact_id: str,
    address: Optional[str],
    capture_pos_K: int,
    capture_pos_V: int,
) -> None:
    """Two-pass capture: K@pos_K, V@pos_V, spliced into one slot.

    If ``capture_pos_K == capture_pos_V`` this reduces to a single call to
    ``write_fact`` (bit-equal).
    """
    if int(capture_pos_K) == int(capture_pos_V):
        write_fact(patcher, bank, tokenizer,
                   write_prompt=write_prompt, fact_id=fact_id,
                   address=address, capture_pos=int(capture_pos_K))
        return

    # Pass 1 — K at pos_K (we also write V@pos_K but throw it away).
    scratch_K = fresh_bank(patcher.model)
    scratch_K.value_scale_mode = bank.value_scale_mode
    scratch_K.value_target_rms = bank.value_target_rms
    scratch_K.bank_key_mode = bank.bank_key_mode
    write_fact(patcher, scratch_K, tokenizer,
               write_prompt=write_prompt, fact_id=fact_id,
               address=address, capture_pos=int(capture_pos_K))

    # Pass 2 — V at pos_V (K@pos_V discarded).
    scratch_V = fresh_bank(patcher.model)
    scratch_V.value_scale_mode = bank.value_scale_mode
    scratch_V.value_target_rms = bank.value_target_rms
    scratch_V.bank_key_mode = bank.bank_key_mode
    write_fact(patcher, scratch_V, tokenizer,
               write_prompt=write_prompt, fact_id=fact_id,
               address=address, capture_pos=int(capture_pos_V))

    # Splice: K from scratch_K, V from scratch_V -> bank (one new slot).
    K_per_layer = [scratch_K.M_K[layer][-1].clone() for layer in range(scratch_K.num_layers)]
    V_per_layer = [scratch_V.M_V[layer][-1].clone() for layer in range(scratch_V.num_layers)]
    patcher._apply_smart_flags(bank)
    bank.append(K_per_layer, V_per_layer, fact_id=fact_id, address=address)
