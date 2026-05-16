"""Public Interrupt API for HNM (Hippocampus-style Native Memory).

The user idea: at any (round, layer, position) during inference, an external
caller (human or program) can inject a hidden vector ``h_inject`` into the
AttentionBank, where it becomes part of the next round's same-layer K/V via
the projector.

This module exposes a small, stable function set; all bookkeeping lives in
the bank itself (see ``v2/core/attention_bank.py``).

Usage::

    from v2.core import AttentionBank
    from v2.core.interrupt_api import interrupt

    bank = AttentionBank(num_layers=L, hidden_size=d, device=dev)
    # ... after round 1, before round 2:
    interrupt(bank, layer=9, h=my_h_vec, position=last_token_idx, round_idx=1)
    # round 2 will see this entry in layer 9
"""
from __future__ import annotations

import torch


def interrupt(bank, *, layer: int, h: torch.Tensor, position: int = -1,
              round_idx: int = 0, batch_idx: int = 0) -> None:
    """Inject a single hidden vector into the bank at ``layer``.

    Args:
        bank: AttentionBank instance.
        layer: target decoder layer index.
        h: hidden vector of shape [d] or [1, d].
        position: source position tag (for debugging).
        round_idx: source round tag.
        batch_idx: source batch index tag.
    """
    if h.dim() == 1:
        h = h.unsqueeze(0)
    assert h.shape[-1] == bank.hidden_size, "h dim must match bank.hidden_size"
    if bank.frozen:
        # caller must un-freeze; we don't auto-toggle
        raise RuntimeError("bank is frozen; un-freeze before interrupt()")
    bank.write(layer=layer, h_in_at_paused=h.to(bank.device),
               positions=[(batch_idx, position)], round_idx=round_idx)


def interrupt_batch(bank, *, layer: int, hs: torch.Tensor,
                    round_idx: int = 0) -> None:
    """Inject a batch of hidden vectors at the same layer.

    Args:
        hs: tensor of shape [N, d].
    """
    assert hs.dim() == 2 and hs.shape[-1] == bank.hidden_size
    if bank.frozen:
        raise RuntimeError("bank is frozen; un-freeze before interrupt_batch()")
    bank.write(layer=layer, h_in_at_paused=hs.to(bank.device),
               positions=[(0, -1)] * hs.shape[0], round_idx=round_idx)


def preload_long_term(bank, *, layer: int, hs: torch.Tensor,
                       freeze_after: bool = True) -> None:
    """Bulk-load long-term memory (e.g., Exp35b b-vectors) at one layer."""
    bank.frozen = False
    bank.slots[layer] = hs.to(device=bank.device, dtype=bank.slots[layer].dtype)
    bank.tags[layer] = [(0, -1)] * hs.shape[0]
    if freeze_after:
        bank.frozen = True
