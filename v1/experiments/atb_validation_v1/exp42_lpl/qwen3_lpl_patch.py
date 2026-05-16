"""Monkey-patch Qwen3Attention + Qwen3DecoderLayer for LPL.

Design:
- We attach an ``lpl_state`` object to the model (model.lpl_state). When
  this is None or its ``enabled`` flag is False, the patched forwards behave
  exactly like the originals → Gate 0 bit-equal.
- DecoderLayer wrapper: (1) computes p_pause from h_in (post-input_layernorm),
  (2) calls original sub-attn + mlp, (3) at paused positions overwrites the
  layer output with residual (skip), (4) writes h_in (pre-layernorm) into the
  AttentionBank for paused positions.
- Attention wrapper: if bank for layer is non-empty AND round_idx >= 2,
  project bank h through self.k_proj/k_norm and self.v_proj, concat to
  K_self / V_self (bank slice gets NO RoPE) and run eager attention. The
  per-position bank_gate scales bank V columns before concat.

NOTE: We only support eager attention (``attn_implementation="eager"``).
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import original symbols from transformers
from transformers.models.qwen3 import modeling_qwen3 as _qwen3
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .attention_bank import AttentionBank, LPLHeads


# ---------------------------------------------------------------------------
# Shared per-forward state. Set/cleared by LPLRuntime around each round.

class LPLState:
    """Container attached to model.lpl_state during LPL runs."""

    def __init__(
        self,
        bank: AttentionBank,
        heads: LPLHeads | None,
        *,
        round_idx: int = 1,
        enabled: bool = True,
        force_pause_mask: torch.Tensor | None = None,
    ):
        self.bank = bank
        self.heads = heads  # may be None for "no heads" mode (pure pass-through, fully disabled)
        self.round_idx = round_idx
        self.enabled = enabled
        # Optional [B, T] bool override for the pause decision (Phase A frozen mode).
        self.force_pause_mask = force_pause_mask
        # Diagnostic counters
        self.pause_count_per_layer: list[int] = [0] * bank.num_layers
        # Captured by the last DecoderLayer wrapper so the runtime can apply halt head
        self.last_layer_hidden: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Patched attention forward.

def _make_lpl_attention_forward(orig_forward):
    def lpl_attention_forward(
        self: Qwen3Attention,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        # Locate the global LPL state.
        state: LPLState | None = getattr(self, "_lpl_state_ref", lambda: None)()
        # Disabled → exact original path.
        if state is None or not state.enabled:
            return orig_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        layer = self.layer_idx
        bank_h = state.bank.read(layer)
        # No bank entries or first round → original path.
        if bank_h is None or state.round_idx < 2:
            return orig_forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        # === Bank-augmented path (round ≥ 2 and bank non-empty) =============
        input_shape = hidden_states.shape[:-1]  # [B, T]
        B, T = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        # --- Bank projection (no RoPE on bank K) -----------------------------
        # bank_h: [N_b, d_model]
        Nb = bank_h.shape[0]
        # match attention dtype/device
        bh = bank_h.to(device=hidden_states.device, dtype=hidden_states.dtype)
        # project through current layer's k/v projection (with k_norm), no rotary
        k_bank = self.k_norm(
            self.k_proj(bh).view(Nb, -1, self.head_dim)
        )  # [N_b, n_kv_heads, d_h]
        v_bank = self.v_proj(bh).view(Nb, -1, self.head_dim)  # [N_b, n_kv_heads, d_h]
        # add batch dim and put seq dim at axis=2 → [B, n_kv_heads, N_b, d_h]
        n_kv = k_bank.shape[1]
        k_bank = k_bank.unsqueeze(0).expand(B, Nb, n_kv, self.head_dim).transpose(1, 2)
        v_bank = v_bank.unsqueeze(0).expand(B, Nb, n_kv, self.head_dim).transpose(1, 2)

        # --- per-position bank gate on V ------------------------------------
        if state.heads is not None:
            bgate = state.heads.bank_gate_heads[layer](hidden_states)  # [B, T, 1]
        else:
            bgate = torch.ones(B, T, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        # bgate must broadcast on the per-query weighting of V_bank columns —
        # cleanest is to apply it after the softmax (weights @ V_bank scaled).
        # For simplicity here we modulate V_bank then compute combined softmax;
        # since attn_weights at column j of bank == w_j, the effective
        # contribution becomes bgate_q * w_j * v_bank_j when applied per-query
        # — that needs post-softmax mul. We do it post-softmax below.

        # --- Combined eager attention ---------------------------------------
        # Concat keys/values along seq axis (dim=2) — repeat_kv equalizes
        # the head count to num_attention_heads first for clean matmul.
        K_self = repeat_kv(key_states, self.num_key_value_groups)
        V_self = repeat_kv(value_states, self.num_key_value_groups)
        K_bank_full = repeat_kv(k_bank, self.num_key_value_groups)
        V_bank_full = repeat_kv(v_bank, self.num_key_value_groups)

        K_aug = torch.cat([K_self, K_bank_full], dim=2)  # [B, H, T+Nb, d_h]
        V_aug = torch.cat([V_self, V_bank_full], dim=2)

        attn_scores = torch.matmul(query_states, K_aug.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            # attention_mask is [B, 1, T_q, T_kv_self]; we need to pad columns
            # for the bank slice with 0 (all visible).
            mask_self = attention_mask  # [B, 1, T, T]
            T_q = mask_self.shape[-2]
            T_kv_self = mask_self.shape[-1]
            mask_bank = torch.zeros(
                mask_self.shape[0],
                mask_self.shape[1],
                T_q,
                Nb,
                device=mask_self.device,
                dtype=mask_self.dtype,
            )
            mask_full = torch.cat([mask_self, mask_bank], dim=-1)
            attn_scores = attn_scores + mask_full

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply bank gate post-softmax: only weights on bank columns are
        # multiplied by per-query bgate. Self columns untouched.
        # attn_weights: [B, H, T_q, T+Nb]; bgate: [B, T_q, 1] → unsqueeze head dim.
        bgate_bh = bgate.unsqueeze(1)  # [B, 1, T_q, 1]
        # Build a multiplier of shape [B, 1, T_q, T+Nb] that is 1 on self, bgate on bank.
        T_kv_self = attn_weights.shape[-1] - Nb
        ones_self = torch.ones(
            B, 1, attn_weights.shape[-2], T_kv_self,
            device=attn_weights.device, dtype=attn_weights.dtype,
        )
        gate_bank = bgate_bh.expand(B, 1, attn_weights.shape[-2], Nb).to(attn_weights.dtype)
        gate_full = torch.cat([ones_self, gate_bank], dim=-1)
        attn_weights = attn_weights * gate_full

        attn_output = torch.matmul(attn_weights, V_aug)  # [B, H, T_q, d_h]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return lpl_attention_forward


# ---------------------------------------------------------------------------
# Patched decoder layer forward.

def _make_lpl_decoder_forward(orig_forward):
    def lpl_decoder_forward(
        self: Qwen3DecoderLayer,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        state: LPLState | None = getattr(self, "_lpl_state_ref", lambda: None)()
        if state is None or not state.enabled:
            return orig_forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        layer = getattr(self.self_attn, "layer_idx", None)

        # --- Decide pause mask BEFORE the layer runs ------------------------
        # Use h_in (pre-layernorm) — same vector that gets written to bank.
        h_in_pre = hidden_states  # [B, T, d]
        B, T, _ = h_in_pre.shape
        if state.force_pause_mask is not None:
            fpm = state.force_pause_mask
            if callable(fpm):
                pause_mask = fpm(layer, state.round_idx, h_in_pre)  # [B,T] bool or None
                if pause_mask is None:
                    pause_mask = torch.zeros(B, T, dtype=torch.bool, device=h_in_pre.device)
            elif isinstance(fpm, dict):
                pause_mask = fpm.get(layer)
                if pause_mask is None:
                    pause_mask = torch.zeros(B, T, dtype=torch.bool, device=h_in_pre.device)
            else:
                pause_mask = fpm  # [B,T] bool, same for every layer
        elif state.heads is not None:
            p_pause = state.heads.pause_heads[layer](h_in_pre).squeeze(-1)  # [B, T]
            pause_mask = p_pause > 0.5
        else:
            pause_mask = torch.zeros(B, T, dtype=torch.bool, device=h_in_pre.device)

        # --- Run original layer (attention + mlp + residuals) ---------------
        out = orig_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # --- Overwrite paused positions with residual (= skip whole layer) --
        if pause_mask.any():
            # Capture paused-position h vectors (pre-layernorm) and write to bank.
            idx = pause_mask.nonzero(as_tuple=False)  # [N_p, 2] (batch, pos)
            paused_h = h_in_pre[pause_mask]  # [N_p, d]
            # Overwrite layer output at paused positions
            out = out.clone()
            out[pause_mask] = h_in_pre[pause_mask]
            # Write to bank
            positions = [(int(idx[i, 0]), int(idx[i, 1])) for i in range(idx.shape[0])]
            state.bank.write(layer, paused_h, positions, state.round_idx)
            state.pause_count_per_layer[layer] += idx.shape[0]

        # Capture last-layer hidden so runtime can apply halt head later.
        # We do this for every layer; the last assignment wins.
        state.last_layer_hidden = out
        return out

    return lpl_decoder_forward


# ---------------------------------------------------------------------------
# Install / uninstall.

_PATCH_MARK = "_lpl_patched"


def install_lpl_patch(model, state: LPLState | None = None) -> None:
    """Monkey-patch every Qwen3DecoderLayer and its Qwen3Attention in ``model``.

    If ``state`` is given, attach a weakref-like getter so the patched forwards
    can find it. Caller (LPLRuntime) is expected to set/update model.lpl_state
    before each round.
    """
    if getattr(model, _PATCH_MARK, False):
        # Already patched; just refresh state ref binding.
        if state is not None:
            model.lpl_state = state
        return

    # Find all decoder layers (work for Qwen3ForCausalLM or Qwen3Model)
    base_model = model.model if hasattr(model, "model") else model
    layers = base_model.layers  # ModuleList of Qwen3DecoderLayer

    # Save originals (unbound class methods).
    orig_decoder_forward = Qwen3DecoderLayer.forward
    orig_attention_forward = Qwen3Attention.forward
    model._lpl_orig_decoder_forward = orig_decoder_forward
    model._lpl_orig_attention_forward = orig_attention_forward

    new_decoder_fwd = _make_lpl_decoder_forward(orig_decoder_forward)
    new_attn_fwd = _make_lpl_attention_forward(orig_attention_forward)

    # Bind per-instance so we don't globally clobber Qwen3 classes.
    import types

    for layer in layers:
        layer.forward = types.MethodType(new_decoder_fwd, layer)
        layer.self_attn.forward = types.MethodType(new_attn_fwd, layer.self_attn)
        # Each module needs a callable that returns the current state.
        layer._lpl_state_ref = lambda m=model: getattr(m, "lpl_state", None)
        layer.self_attn._lpl_state_ref = lambda m=model: getattr(m, "lpl_state", None)

    model.lpl_state = state
    setattr(model, _PATCH_MARK, True)


def uninstall_lpl_patch(model) -> None:
    if not getattr(model, _PATCH_MARK, False):
        return
    base_model = model.model if hasattr(model, "model") else model
    import types

    for layer in base_model.layers:
        # Restore class-level forwards by deleting per-instance bindings.
        # (the bound MethodType references will be garbage-collected.)
        layer.forward = types.MethodType(model._lpl_orig_decoder_forward, layer)
        layer.self_attn.forward = types.MethodType(
            model._lpl_orig_attention_forward, layer.self_attn
        )
        if hasattr(layer, "_lpl_state_ref"):
            del layer._lpl_state_ref
        if hasattr(layer.self_attn, "_lpl_state_ref"):
            del layer.self_attn._lpl_state_ref

    model.lpl_state = None
    setattr(model, _PATCH_MARK, False)


@contextmanager
def lpl_state_scope(model, state: LPLState):
    """Temporarily set ``model.lpl_state``; restore prev on exit."""
    prev = getattr(model, "lpl_state", None)
    model.lpl_state = state
    try:
        yield
    finally:
        model.lpl_state = prev
