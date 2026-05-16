"""Single-slot AttentionBank patch for Gemma3 architecture (e21 protocol).

Gemma3 is closer to Qwen3 (has q_norm/k_norm), but with sliding_window per layer
and `position_embeddings=(cos,sin)` passed in. No softcap on attn logits.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gemma3 import modeling_gemma3 as _gemma3
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3Attention, apply_rotary_pos_emb, repeat_kv,
)

from .attention_bank import AttentionBank, LPLHeads


class LPLState:
    def __init__(self, bank: AttentionBank, heads: LPLHeads | None, *,
                 round_idx: int = 1, enabled: bool = True,
                 force_pause_mask=None):
        self.bank = bank
        self.heads = heads
        self.round_idx = round_idx
        self.enabled = enabled
        self.force_pause_mask = force_pause_mask


@contextmanager
def lpl_state_scope(model, state):
    prev = getattr(model, "lpl_state", None)
    model.lpl_state = state
    try:
        yield
    finally:
        model.lpl_state = prev


def _make_bank_attn_forward(orig_forward):
    def fwd(self: Gemma3Attention, hidden_states, position_embeddings,
            attention_mask=None, past_key_values=None, **kwargs):
        state = getattr(self, "_lpl_state_ref", lambda: None)()
        if state is None or not state.enabled:
            return orig_forward(self, hidden_states=hidden_states,
                                position_embeddings=position_embeddings,
                                attention_mask=attention_mask,
                                past_key_values=past_key_values, **kwargs)

        layer = self.layer_idx
        bank_h = state.bank.read(layer)
        if bank_h is None or state.round_idx < 2:
            return orig_forward(self, hidden_states=hidden_states,
                                position_embeddings=position_embeddings,
                                attention_mask=attention_mask,
                                past_key_values=past_key_values, **kwargs)

        input_shape = hidden_states.shape[:-1]
        B, T = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        Nb = bank_h.shape[0]
        bh = bank_h.to(device=hidden_states.device, dtype=hidden_states.dtype)
        k_bank = self.k_proj(bh).view(Nb, -1, self.head_dim)
        v_bank = self.v_proj(bh).view(Nb, -1, self.head_dim)
        # apply k_norm to bank keys (consistent with self-K), no RoPE
        k_bank = self.k_norm(k_bank.unsqueeze(0)).squeeze(0)
        n_kv = k_bank.shape[1]
        k_bank = k_bank.unsqueeze(0).expand(B, Nb, n_kv, self.head_dim).transpose(1, 2)
        v_bank = v_bank.unsqueeze(0).expand(B, Nb, n_kv, self.head_dim).transpose(1, 2)

        K_self = repeat_kv(k, self.num_key_value_groups)
        V_self = repeat_kv(v, self.num_key_value_groups)
        K_bank_full = repeat_kv(k_bank, self.num_key_value_groups)
        V_bank_full = repeat_kv(v_bank, self.num_key_value_groups)

        K_aug = torch.cat([K_self, K_bank_full], dim=2)
        V_aug = torch.cat([V_self, V_bank_full], dim=2)

        scaling = getattr(self, "scaling", self.head_dim ** -0.5)

        scores = torch.matmul(q, K_aug.transpose(2, 3)) * scaling

        if attention_mask is not None:
            mask_self = attention_mask
            # gemma3 mask shape varies; coerce to [B, 1, T_q, T_k]
            if mask_self.dim() == 4:
                T_q = mask_self.shape[-2]
                mask_bank = torch.zeros(
                    mask_self.shape[0], mask_self.shape[1], T_q, Nb,
                    device=mask_self.device, dtype=mask_self.dtype,
                )
                mask_full = torch.cat([mask_self, mask_bank], dim=-1)
                scores = scores + mask_full

        attn_w = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        if state.heads is not None:
            bgate = state.heads.bank_gate_heads[layer](hidden_states)
        else:
            bgate = torch.ones(B, T, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        T_kv_self = attn_w.shape[-1] - Nb
        ones_self = torch.ones(B, 1, attn_w.shape[-2], T_kv_self,
                               device=attn_w.device, dtype=attn_w.dtype)
        gate_bank = bgate.unsqueeze(1).expand(B, 1, attn_w.shape[-2], Nb).to(attn_w.dtype)
        gate_full = torch.cat([ones_self, gate_bank], dim=-1)
        attn_w = attn_w * gate_full

        out = torch.matmul(attn_w, V_aug)
        out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        out = self.o_proj(out)
        return out, attn_w

    return fwd


def install_lpl_patch(model):
    """Install Gemma3 single-slot bank patch."""
    orig = Gemma3Attention.forward
    model._lpl_orig_attention_forward = orig
    new_fwd = _make_bank_attn_forward(orig)

    def state_ref(layer):
        return lambda: getattr(model, "lpl_state", None)

    for layer in model.model.layers:
        attn = layer.self_attn
        attn._lpl_state_ref = state_ref(attn.layer_idx)
        attn.forward = new_fwd.__get__(attn, type(attn))
