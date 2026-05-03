"""AttentionNative DeltaMemory (Stage 13A).

Replaces v1's stitched pipeline (encoder -> KeyProjector -> bank -> broadcast
residual injection) with a single linear-algebra primitive: per-layer K/V
concatenation inside attention itself.

Mathematical statement
----------------------
For each non-shared attention layer l of a frozen transformer:

    Attn_l(Q, K, V)  -->  Attn_l( Q,  [K; M_K^(l)],  [V; alpha * M_V^(l)] )

where (M_K^(l), M_V^(l)) are stored once per fact during a single forward pass
of the write-prompt. The bank carries no learnable parameters at all; the
"retrieval space" *is* the model's native K-space, and the attention softmax
is the contrastive engine.

RoPE handling (position-agnostic, Stage 13A user-approved option B)
-------------------------------------------------------------------
The bank stores **pre-RoPE post-norm** K (and post-norm V).  At attention
time we keep a pre-RoPE copy of Q and use it only for the bank slice:

    scores_orig = q_post @ k_post^T * scaling          # standard, with RoPE
    scores_bank = q_pre  @ M_K^T    * scaling          # both sides pre-RoPE
    scores       = cat([scores_orig + mask, scores_bank], dim=-1)
    weights      = softmax(scores, dim=-1)
    out          = weights[..., :T] @ V_orig
                 + weights[..., T:] @ (alpha * M_V_repeated)

Concretely each query at position t sees bank slot k via the *position-free*
inner product q_pre · M_K[k], identical for every t.  This is the user's
stated requirement: "make the bank a pure semantic pool".

GQA handling
------------
Bank tensors are stored at the model's `num_key_value_heads` resolution (e.g.,
1 for Gemma-4-E2B's MQA, 8 for DeepSeek-V2-Lite, etc.) and expanded with the
exact same `repeat_kv` path as the original K/V.

KV-shared layers
----------------
Layers with ``is_kv_shared_layer=True`` (Gemma 4 KV-sharing scheme) reuse
their source layer's K/V at write time and consult the source layer's bank
slot at read time, so every attention layer sees the bank — non-shared
layers via their own slot, shared layers via ``kv_shared_layer_index``.

Bit-equal sanity (Gate 13A.1)
-----------------------------
When ``bank.empty == True`` the patched forward must produce logits identical
to the unpatched model. The unit test in ``tests/test_attn_native_bank.py``
asserts max-abs-diff < 1e-3 on the full 35-layer Gemma-4-E2B in bf16; in
practice the observed value is exactly 0.0 on both MPS and CUDA.

This module is intentionally minimal: ~250 LoC including docs.  No 1500-step
training, no encoder, no projector.  Just K/V concat.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------

@dataclass
class AttnNativeBank:
    """Per-layer K/V slot store.

    Tensors live on the same device as the host model.  All slots are inserted
    via :meth:`write_facts`, which performs a single forward pass over each
    fact's write-prompt and captures `(K_pre_rope, V)` at the configured
    address position (default: last real token).

    Note: Gemma-4 uses per-layer head_dim (sliding=256, full=512).  We store
    one head_dim per layer in ``head_dims`` so the bank works on every layer.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int                       # default / fallback
    head_dims: list[int] = field(default_factory=list)
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.bfloat16

    # populated lazily; kept on the model's device
    M_K: list[torch.Tensor] = field(default_factory=list)
    M_V: list[torch.Tensor] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    address_strs: list[str] = field(default_factory=list)
    _shared_to_source: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.head_dims:
            self.head_dims = [self.head_dim] * self.num_layers
        if not self.M_K:
            for layer in range(self.num_layers):
                d = self.head_dims[layer]
                self.M_K.append(torch.empty(0, self.num_kv_heads, d,
                                            device=self.device, dtype=self.dtype))
                self.M_V.append(torch.empty(0, self.num_kv_heads, d,
                                            device=self.device, dtype=self.dtype))

    @property
    def empty(self) -> bool:
        return self.size == 0

    @property
    def size(self) -> int:
        return self.M_K[0].size(0) if self.M_K else 0

    def clear(self) -> None:
        for layer in range(self.num_layers):
            d = self.head_dims[layer]
            self.M_K[layer] = torch.empty(0, self.num_kv_heads, d,
                                          device=self.device, dtype=self.dtype)
            self.M_V[layer] = torch.empty(0, self.num_kv_heads, d,
                                          device=self.device, dtype=self.dtype)
        self.fact_ids.clear()
        self.address_strs.clear()

    # -------- write path --------

    def append(
        self,
        per_layer_K: list[torch.Tensor],
        per_layer_V: list[torch.Tensor],
        fact_id: str,
        address: str,
    ) -> None:
        """Append one fact across all layers.

        Each tensor is shape ``[num_kv_heads, head_dim_layer]``.

        .. note::
            Single-fact append performs a per-layer ``torch.cat`` and is
            therefore O(N) per call (O(N²) for N writes).  For bulk ingest
            of many facts use :meth:`bulk_append` which concatenates once
            per layer.
        """
        if len(per_layer_K) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer K, got {len(per_layer_K)}")
        for layer in range(self.num_layers):
            d = self.head_dims[layer]
            k = per_layer_K[layer].to(self.device, self.dtype).reshape(1, self.num_kv_heads, d)
            v = per_layer_V[layer].to(self.device, self.dtype).reshape(1, self.num_kv_heads, d)
            self.M_K[layer] = torch.cat([self.M_K[layer], k], dim=0)
            self.M_V[layer] = torch.cat([self.M_V[layer], v], dim=0)
        self.fact_ids.append(fact_id)
        self.address_strs.append(address)

    def bulk_append(
        self,
        per_layer_K_batches: list[torch.Tensor],
        per_layer_V_batches: list[torch.Tensor],
        fact_ids: list[str],
        addresses: list[str],
    ) -> None:
        """Append a batch of N facts in O(N) total — single ``cat`` per layer.

        Each ``per_layer_K_batches[layer]`` must be ``[N, num_kv_heads, head_dim_layer]``.
        """
        if len(per_layer_K_batches) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer K, got {len(per_layer_K_batches)}")
        n = len(fact_ids)
        for layer in range(self.num_layers):
            d = self.head_dims[layer]
            k = per_layer_K_batches[layer].to(self.device, self.dtype).reshape(n, self.num_kv_heads, d)
            v = per_layer_V_batches[layer].to(self.device, self.dtype).reshape(n, self.num_kv_heads, d)
            self.M_K[layer] = torch.cat([self.M_K[layer], k], dim=0)
            self.M_V[layer] = torch.cat([self.M_V[layer], v], dim=0)
        self.fact_ids.extend(fact_ids)
        self.address_strs.extend(addresses)

    def state_dict(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "head_dims": list(self.head_dims),
            "M_K": [t.cpu() for t in self.M_K],
            "M_V": [t.cpu() for t in self.M_V],
            "fact_ids": list(self.fact_ids),
            "address_strs": list(self.address_strs),
        }

    @classmethod
    def from_state_dict(cls, sd: dict, device="cpu", dtype=torch.bfloat16) -> "AttnNativeBank":
        bank = cls(num_layers=sd["num_layers"],
                   num_kv_heads=sd["num_kv_heads"],
                   head_dim=sd["head_dim"],
                   head_dims=list(sd.get("head_dims") or [sd["head_dim"]] * sd["num_layers"]),
                   device=device, dtype=dtype)
        bank.M_K = [t.to(device, dtype) for t in sd["M_K"]]
        bank.M_V = [t.to(device, dtype) for t in sd["M_V"]]
        bank.fact_ids = list(sd["fact_ids"])
        bank.address_strs = list(sd["address_strs"])
        return bank


# ---------------------------------------------------------------------------
# Patched forward
# ---------------------------------------------------------------------------

def _make_patched_forward(orig_forward, layer_idx: int, ctx: "AttnNativePatcher"):
    """Build the per-layer monkey-patched forward.

    Mirrors :class:`Gemma3nTextAttention.forward` but:
      * keeps a pre-RoPE Q copy alongside the post-RoPE Q for bank scoring,
      * captures the layer's pre-RoPE K (and V) when ``ctx.capture_mode`` is on,
      * concatenates bank K/V into the attention computation when
        ``ctx.bank`` is attached, ``ctx.alpha > 0``, and the bank is non-empty.
    """
    from transformers.models.gemma4.modeling_gemma4 import (
        apply_rotary_pos_emb, repeat_kv,
    )

    def forward(self, hidden_states, position_embeddings,
                attention_mask=None, past_key_values=None,
                shared_kv_states=None, **kwargs):
        bank = ctx.bank
        alpha = ctx.alpha
        capture = ctx.capture_mode
        capture_pos = ctx.capture_pos  # int or None; default = last token

        input_shape = hidden_states.shape[:-1]
        head_dim = getattr(self, "head_dim", None) or self.config.head_dim
        hidden_shape = (*input_shape, -1, head_dim)
        cos, sin = position_embeddings

        # --- Q (keep pre-RoPE for bank scoring) ---
        q_pre = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        q_post = apply_rotary_pos_emb(q_pre, cos, sin, unsqueeze_dim=2)
        q_post = q_post.transpose(1, 2)            # [B, Hq, T, d]
        q_pre = q_pre.transpose(1, 2)              # [B, Hq, T, d]

        # --- K, V (post-norm, pre-RoPE captured for bank write) ---
        if self.is_kv_shared_layer:
            # transformers 5.7 passes shared_kv_states as kwarg; 5.5 moved it
            # to past_key_values.shared_layers.  Fall back across both APIs.
            shared_dict = shared_kv_states
            if shared_dict is None and past_key_values is not None:
                shared_dict = getattr(past_key_values, "shared_layers", None)
            if shared_dict is None or self.kv_shared_layer_index not in shared_dict:
                # Last-resort recompute (should not happen in eager prefill,
                # but keeps the patcher robust to future API drift).
                k_pre = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
                v_post_norm = self.v_norm(self.v_proj(hidden_states).view(hidden_shape))
                key_states = apply_rotary_pos_emb(k_pre, cos, sin, unsqueeze_dim=2).transpose(1, 2)
                value_states = v_post_norm.transpose(1, 2)
            else:
                key_states, value_states = shared_dict[self.kv_shared_layer_index]
                key_states = key_states.to(q_post.device)
                value_states = value_states.to(q_post.device)
            k_pre_for_capture = None  # do not capture on shared layers
        else:
            k_pre = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))   # [B, T, Hkv, d]
            v_post_norm = self.v_norm(self.v_proj(hidden_states).view(hidden_shape))
            k_post = apply_rotary_pos_emb(k_pre, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value_states = v_post_norm.transpose(1, 2)
            key_states = k_post
            k_pre_for_capture = k_pre  # [B, T, Hkv, d]

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if self.store_full_length_kv:
            if shared_kv_states is not None:
                shared_kv_states[self.layer_idx] = key_states, value_states
            elif past_key_values is not None:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        # --- capture for bank write (single fact at a time) ---
        if capture and k_pre_for_capture is not None:
            T_full = k_pre_for_capture.size(1)
            pos = (T_full - 1) if capture_pos is None else int(capture_pos)
            ctx._capture_K[layer_idx] = k_pre_for_capture[:, pos, :, :].detach()  # [B, Hkv, d]
            ctx._capture_V[layer_idx] = (value_states.transpose(1, 2)            # [B, T, Hkv, d]
                                          [:, pos, :, :].detach())

        # --- standard attention ---
        scaling = getattr(self, "scaling", None) or (head_dim ** -0.5)
        # NOTE: sliding-window masking is already baked into ``attention_mask``
        # by HF's eager preparation, so we do not re-apply it here.

        k_repeat = repeat_kv(key_states, self.num_key_value_groups)
        v_repeat = repeat_kv(value_states, self.num_key_value_groups)
        scores_orig = torch.matmul(q_post, k_repeat.transpose(2, 3)) * scaling  # [B,Hq,T,Tk]
        if attention_mask is not None:
            scores_orig = scores_orig + attention_mask[..., : scores_orig.size(-1)]

        # --- bank attention (position-agnostic, GQA-aware).
        # KV-shared layers re-use the bank slot from their source layer so the
        # injection is visible in every attention layer, not only non-shared
        # ones (otherwise 20/35 layers in Gemma-4-E2B would skip the bank).
        bank_layer_idx = (
            getattr(self, "kv_shared_layer_index", layer_idx)
            if self.is_kv_shared_layer else layer_idx
        )
        do_inject = (
            bank is not None
            and not bank.empty
            and alpha > 0.0
            and bank_layer_idx is not None
            and bank.M_K[bank_layer_idx].size(0) > 0
        )
        if do_inject:
            mk = bank.M_K[bank_layer_idx].to(q_pre.dtype).to(q_pre.device)  # [N, Hkv, d]
            mv = bank.M_V[bank_layer_idx].to(q_pre.dtype).to(q_pre.device)
            mk_e = repeat_kv(mk.unsqueeze(0).transpose(1, 2), self.num_key_value_groups)  # [1, Hq, N, d]
            mv_e = repeat_kv(mv.unsqueeze(0).transpose(1, 2), self.num_key_value_groups)
            mk_e = mk_e.expand(q_pre.size(0), -1, -1, -1)
            mv_e = mv_e.expand(q_pre.size(0), -1, -1, -1)
            scores_bank = torch.matmul(q_pre, mk_e.transpose(2, 3)) * scaling  # [B,Hq,T,N]
            scores = torch.cat([scores_orig, scores_bank], dim=-1)
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_post.dtype)
            T_orig = scores_orig.size(-1)
            out_orig = torch.matmul(weights[..., :T_orig], v_repeat)
            out_bank = torch.matmul(weights[..., T_orig:], alpha * mv_e)
            attn_out = (out_orig + out_bank).transpose(1, 2).contiguous()
        else:
            weights = F.softmax(scores_orig, dim=-1, dtype=torch.float32).to(q_post.dtype)
            attn_out = torch.matmul(weights, v_repeat).transpose(1, 2).contiguous()

        attn_out = attn_out.reshape(*input_shape, -1)
        return self.o_proj(attn_out), None

    return forward


# ---------------------------------------------------------------------------
# Patcher (context manager)
# ---------------------------------------------------------------------------

class AttnNativePatcher:
    """Monkey-patches every Gemma3nTextAttention.forward in `model.model.layers`.

    Two modes:
      * ``with patcher.capturing(layer_pos=None)``: forward writes the bank's
        per-layer K/V tensors (pre-RoPE) into ``patcher._capture_K/_V``.
      * ``with patcher.injecting(bank=bank, alpha=1.0)``: subsequent forwards
        use the bank.

    Use :func:`write_fact` and :func:`forward_with_bank` for the standard
    path; the context managers are reusable building blocks.
    """

    def __init__(self, model):
        self.model = model
        # Resolve attention modules.  Path varies by family:
        #  * Gemma4ForConditionalGeneration   -> model.model.language_model.layers
        #  * Gemma3 / Llama / Qwen / DeepSeek -> model.model.layers
        #  * sometimes wrapped one extra time -> model.language_model.model.layers
        layers = None
        for path in ("model.model.language_model.layers",
                     "model.model.layers",
                     "model.language_model.model.layers",
                     "model.language_model.layers",
                     "language_model.layers",
                     "model.layers"):
            obj = model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok and hasattr(obj, "__len__") and len(obj) > 0 and hasattr(obj[0], "self_attn"):
                layers = obj
                break
        if layers is None:
            raise RuntimeError("AttnNativePatcher: could not locate decoder layers on the model")
        self.attn_modules: list[nn.Module] = [layer.self_attn for layer in layers]
        self.num_layers = len(self.attn_modules)

        # Reject non-Gemma4 attention modules: the patched forward references
        # Gemma-4-specific attributes (q_norm, k_norm, v_norm, is_kv_shared_layer,
        # kv_shared_layer_index, store_full_length_kv) and would error on the
        # first forward pass with Llama / Qwen / DeepSeek / mock attention.
        cls_name = type(self.attn_modules[0]).__name__
        if "Gemma4" not in cls_name:
            raise NotImplementedError(
                f"AttnNativePatcher currently supports only Gemma-4 attention "
                f"modules (got {cls_name}). Adding support for {cls_name} requires "
                f"a model-specific patched forward; PRs welcome."
            )

        cfg = getattr(model.config, "text_config", model.config)
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim

        self._orig_forwards: list = [None] * self.num_layers
        self.bank: AttnNativeBank | None = None
        self.alpha: float = 0.0
        self.capture_mode: bool = False
        self.capture_pos: int | None = None
        self._capture_K: list[torch.Tensor | None] = [None] * self.num_layers
        self._capture_V: list[torch.Tensor | None] = [None] * self.num_layers

    def install(self) -> None:
        for i, m in enumerate(self.attn_modules):
            if self._orig_forwards[i] is None:
                self._orig_forwards[i] = m.forward
                bound = _make_patched_forward(self._orig_forwards[i], i, self).__get__(m, type(m))
                m.forward = bound

    def remove(self) -> None:
        for i, m in enumerate(self.attn_modules):
            if self._orig_forwards[i] is not None:
                m.forward = self._orig_forwards[i]
                self._orig_forwards[i] = None

    @contextmanager
    def patched(self):
        self.install()
        try:
            yield self
        finally:
            self.remove()

    @contextmanager
    def capturing(self, capture_pos: int | None = None):
        self.capture_mode = True
        self.capture_pos = capture_pos
        self._capture_K = [None] * self.num_layers
        self._capture_V = [None] * self.num_layers
        try:
            yield
        finally:
            self.capture_mode = False

    @contextmanager
    def injecting(self, bank: AttnNativeBank, alpha: float = 1.0):
        prev_bank, prev_alpha = self.bank, self.alpha
        self.bank, self.alpha = bank, float(alpha)
        try:
            yield
        finally:
            self.bank, self.alpha = prev_bank, prev_alpha


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def fresh_bank(model) -> AttnNativeBank:
    cfg = getattr(model.config, "text_config", model.config)
    device = next(model.parameters()).device

    # Discover per-layer head_dim by walking attn modules (Gemma-4 has different
    # head_dim on sliding vs full layers: 256 vs 512).
    patcher_probe = AttnNativePatcher(model)
    head_dims: list[int] = []
    for sa in patcher_probe.attn_modules:
        d = getattr(sa, "head_dim", None) or cfg.head_dim
        head_dims.append(int(d))

    return AttnNativeBank(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        head_dims=head_dims,
        device=device,
        dtype=next(model.parameters()).dtype,
    )


def write_fact(
    patcher: AttnNativePatcher,
    bank: AttnNativeBank,
    tokenizer,
    write_prompt: str,
    fact_id: str,
    address: str,
    capture_pos: int | None = None,
) -> None:
    """Single-shot bank insertion: forward the write_prompt and grab K/V.

    The capture position defaults to the last real token of `write_prompt`.
    """
    device = next(patcher.model.parameters()).device
    enc = tokenizer(write_prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    pos = (am.sum(dim=1).item() - 1) if capture_pos is None else int(capture_pos)
    with patcher.patched(), patcher.capturing(capture_pos=pos), torch.no_grad():
        patcher.model(input_ids=ids, attention_mask=am, use_cache=False)
    K_per_layer = []
    V_per_layer = []
    for layer in range(patcher.num_layers):
        kc = patcher._capture_K[layer]
        vc = patcher._capture_V[layer]
        if kc is None:  # shared-KV layer: copy from source layer
            src = patcher.attn_modules[layer]
            src_idx = getattr(src, "kv_shared_layer_index", layer)
            kc = patcher._capture_K[src_idx]
            vc = patcher._capture_V[src_idx]
        K_per_layer.append(kc[0])  # drop batch
        V_per_layer.append(vc[0])
    bank.append(K_per_layer, V_per_layer, fact_id=fact_id, address=address)


def forward_with_bank(
    patcher: AttnNativePatcher,
    bank: AttnNativeBank,
    tokenizer,
    read_prompt: str,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Forward `read_prompt` with the bank attached. Returns logits[0, -1].

    Use ``alpha=0.0`` to verify bit-equivalence to the unpatched model.
    """
    device = next(patcher.model.parameters()).device
    enc = tokenizer(read_prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    with patcher.patched(), patcher.injecting(bank, alpha=alpha), torch.no_grad():
        out = patcher.model(input_ids=ids, attention_mask=am, use_cache=False)
    last = am.sum(dim=1).item() - 1
    return out.logits[0, last].detach()
