"""AttentionNative Mneme (Stage 13A).

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
from typing import Any, Iterable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bank
# ---------------------------------------------------------------------------

ValueScaleMode = Literal["none", "auto_rms_cap", "rms_cap", "auto_unit_rms", "unit_rms"]


def _scale_bank_value_capture(
    v: torch.Tensor,
    *,
    mode: str,
    has_native_v_norm: bool,
    target_rms: float,
    eps: float,
) -> torch.Tensor:
    """Scale captured bank V so alpha has comparable meaning across families.

    ``auto_rms_cap`` is the production default: Gemma-4 already applies native
    ``v_norm`` and is left untouched; Qwen/Llama/GLM-style families without
    native V normalization have captured bank values capped to a fixed
    per-head RMS.  A cap is deliberately safer than exact normalization: if
    a no-v_norm family already has small V activations, we must not amplify it.
    """
    auto_modes = {"auto_rms_cap", "auto_unit_rms"}
    cap_modes = {"auto_rms_cap", "rms_cap"}
    unit_modes = {"auto_unit_rms", "unit_rms"}
    valid_modes = {"none", *auto_modes, "rms_cap", "unit_rms"}
    if mode == "none" or (mode in auto_modes and has_native_v_norm):
        return v
    if mode not in valid_modes:
        raise ValueError(
            f"value_scale_mode must be one of 'none', 'auto_rms_cap', "
            f"'rms_cap', 'auto_unit_rms', 'unit_rms', got {mode!r}"
        )
    if target_rms <= 0.0:
        raise ValueError(f"value_target_rms must be > 0, got {target_rms!r}")
    if eps <= 0.0:
        raise ValueError(f"value_scale_eps must be > 0, got {eps!r}")
    rms = torch.linalg.vector_norm(v.float(), ord=2, dim=-1, keepdim=True)
    rms = rms / (v.size(-1) ** 0.5)
    scale = float(target_rms) / rms.clamp_min(float(eps))
    if mode in cap_modes:
        scale = torch.clamp(scale, max=1.0)
    elif mode not in unit_modes:
        raise ValueError(f"unhandled value_scale_mode {mode!r}")
    scaled = v.float() * scale
    return scaled.to(v.dtype)

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
    # Stage 14A: optional InfoNCE K-projector (KProjectorBank). Identity-init
    # so attaching an untrained projector remains a no-op.
    k_projector: Any = None
    # Stage 14D: bank-only attention temperature. 1.0 = no-op (bit-equal).
    bank_temperature: float = 1.0
    # Stage 16 (v3.2): mHC spectral shield — bank-columns-only column-norm
    # cap.  When True the post-softmax attention weights have each bank column's
    # total received attention capped at ≤ kappa (default 1.0), bounding spectral
    # amplification of the external-KV channel while leaving native sequence
    # columns bit-for-bit unchanged.  Default False keeps v3.1 behaviour.
    # See ``deltamemory.memory.mhc_shield.shield_attention_weights``.
    mhc_shield: bool = False
    # Stage R (v3.3): Dynamic LOPI configuration + per-call state. Default
    # cfg.enabled=False makes the LOPI wrapper a no-op so the merged-softmax
    # branch keeps its v3.2 byte-equal behavior.  See ``deltamemory.memory.lopi``.
    lopi_cfg: Any = None
    lopi_state: Any = None
    # Stage R-7: bank-side V magnitude calibration.  Families without native
    # v_norm (Qwen/Llama/GLM) may store larger M_V than Gemma, so the same
    # alpha can mean different perturbation energy. ``auto_rms_cap`` keeps
    # Gemma untouched and caps only no-v_norm families at write time.
    value_scale_mode: ValueScaleMode = "auto_rms_cap"
    value_target_rms: float = 0.5
    value_scale_eps: float = 1e-6

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
        # Lazy import to keep top-of-file clean and avoid circulars.
        if self.lopi_cfg is None:
            from deltamemory.memory.lopi import LOPIConfig
            self.lopi_cfg = LOPIConfig()  # enabled=False by default
        if self.lopi_state is None:
            from deltamemory.memory.lopi import LOPIState
            self.lopi_state = LOPIState(num_layers=self.num_layers)

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
        if self.lopi_state is not None and hasattr(self.lopi_state, "reset"):
            self.lopi_state.reset()

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
        if len(per_layer_V) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer V, got {len(per_layer_V)}")
        for layer in range(self.num_layers):
            d = self.head_dims[layer]
            expected = (self.num_kv_heads, d)
            if tuple(per_layer_K[layer].shape) != expected:
                raise ValueError(
                    f"append: layer {layer} K shape mismatch "
                    f"(expected {expected}, got {tuple(per_layer_K[layer].shape)})"
                )
            if tuple(per_layer_V[layer].shape) != expected:
                raise ValueError(
                    f"append: layer {layer} V shape mismatch "
                    f"(expected {expected}, got {tuple(per_layer_V[layer].shape)})"
                )
            k = per_layer_K[layer].to(self.device, self.dtype).unsqueeze(0)
            v = per_layer_V[layer].to(self.device, self.dtype).unsqueeze(0)
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
        if len(per_layer_V_batches) != self.num_layers:
            raise ValueError(f"expected {self.num_layers} layer V, got {len(per_layer_V_batches)}")
        if len(fact_ids) != len(addresses):
            raise ValueError(
                f"bulk_append: fact_ids ({len(fact_ids)}) and addresses "
                f"({len(addresses)}) must have equal length"
            )
        n = len(fact_ids)
        for layer in range(self.num_layers):
            d = self.head_dims[layer]
            for name, src in (("K", per_layer_K_batches[layer]), ("V", per_layer_V_batches[layer])):
                expected = (n, self.num_kv_heads, d)
                if tuple(src.shape) != expected:
                    raise ValueError(
                        f"bulk_append: layer {layer} {name} shape mismatch "
                        f"(expected {expected}, got {tuple(src.shape)})"
                    )
            k = per_layer_K_batches[layer].to(self.device, self.dtype)
            v = per_layer_V_batches[layer].to(self.device, self.dtype)
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
            "bank_temperature": float(self.bank_temperature),
            "mhc_shield": bool(self.mhc_shield),
            "value_scale_mode": str(self.value_scale_mode),
            "value_target_rms": float(self.value_target_rms),
            "value_scale_eps": float(self.value_scale_eps),
            "bank_cosine": bool(getattr(self, "bank_cosine", False)),
            "bank_topk": int(getattr(self, "bank_topk", 0) or 0),
            "bank_separate_softmax": bool(getattr(self, "bank_separate_softmax", False)),
            "bank_merge_beta": float(getattr(self, "bank_merge_beta", 1.0)),
        }

    @classmethod
    def from_state_dict(cls, sd: dict, device="cpu", dtype=torch.bfloat16) -> "AttnNativeBank":
        bank = cls(num_layers=sd["num_layers"],
                   num_kv_heads=sd["num_kv_heads"],
                   head_dim=sd["head_dim"],
                   head_dims=list(sd.get("head_dims") or [sd["head_dim"]] * sd["num_layers"]),
                   device=device, dtype=dtype,
                   bank_temperature=float(sd.get("bank_temperature", 1.0)),
                   mhc_shield=bool(sd.get("mhc_shield", False)),
                   value_scale_mode=sd.get("value_scale_mode", "auto_rms_cap"),
                   value_target_rms=float(sd.get("value_target_rms", 0.5)),
                   value_scale_eps=float(sd.get("value_scale_eps", 1e-6)))
        bank.M_K = [t.to(device, dtype) for t in sd["M_K"]]
        bank.M_V = [t.to(device, dtype) for t in sd["M_V"]]
        bank.fact_ids = list(sd["fact_ids"])
        bank.address_strs = list(sd["address_strs"])
        bank.bank_cosine = bool(sd.get("bank_cosine", False))
        bank.bank_topk = int(sd.get("bank_topk", 0) or 0)
        bank.bank_separate_softmax = bool(sd.get("bank_separate_softmax", False))
        bank.bank_merge_beta = float(sd.get("bank_merge_beta", 1.0))
        return bank

    # ------------------------------------------------------------------
    # Phase S — U-LOPI profile attachment
    # ------------------------------------------------------------------

    def attach_lopi_profile(self, model: Any, tokenizer: Any, *,
                            prompts: Any = None,
                            device: Any = None,
                            dtype: Any = None) -> Any:
        """One-shot cold-start residual profile -> ``self.lopi_state.profile``.

        Forward-only.  No gradient, no nn.Parameter introduced.  After the
        call ``self.lopi_state.profile`` holds a :class:`LOPIProfile` and
        ``self.lopi_cfg.profile_mode == "auto"`` will route the depth
        signal through Z-score space.

        Returns the attached :class:`LOPIProfile`.
        """
        from deltamemory.memory.lopi_profiler import profile_residuals

        prof = profile_residuals(
            model, tokenizer,
            prompts=prompts,
            device=device if device is not None else self.device,
            dtype=dtype,
        )
        if prof.num_layers != self.num_layers:
            # Profile counts include LM hidden_states output (block outputs).
            # If they disagree the bank shape and the model decoder are out
            # of sync; surface immediately rather than silently mismatching
            # mu_arch indices.
            raise ValueError(
                f"profile.num_layers={prof.num_layers} != bank.num_layers="
                f"{self.num_layers}; refusing to attach mismatched profile"
            )
        self.lopi_state.profile = prof
        return prof


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

    The Gemma-4-specific bits (q/k/v norms, RoPE, KV-sharing, repeat_kv) are
    routed through ``ctx.adapter`` so the same forward works on Qwen3 / Llama /
    GLM-4 once their :class:`ArchAdapter` is registered. The Gemma-4 path
    remains bit-equal to the upstream forward (max-abs-diff = 0.0 in the
    13A regression test).
    """
    adapter = ctx.adapter

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
        q_pre = adapter.apply_q_norm(self, self.q_proj(hidden_states).view(hidden_shape))

        # --- K, V (post-norm, pre-RoPE captured for bank write) ---
        is_kv_shared = adapter.is_kv_shared(self)
        if is_kv_shared:
            shared_idx = adapter.kv_shared_index(self)
            shared_dict = shared_kv_states
            if shared_dict is None and past_key_values is not None:
                shared_dict = getattr(past_key_values, "shared_layers", None)
            if shared_dict is None or shared_idx not in shared_dict:
                # Last-resort recompute (should not happen in eager prefill,
                # but keeps the patcher robust to future API drift).
                k_pre = adapter.apply_k_norm(self, self.k_proj(hidden_states).view(hidden_shape))
                v_post_norm = adapter.apply_v_norm(self, self.v_proj(hidden_states).view(hidden_shape))
                q_post_, k_post_ = adapter.apply_rope(q_pre, k_pre, cos, sin)
                key_states = k_post_.transpose(1, 2)
                value_states = v_post_norm.transpose(1, 2)
                q_post = q_post_.transpose(1, 2)
            else:
                key_states, value_states = shared_dict[shared_idx]
                key_states = key_states.to(q_pre.device)
                value_states = value_states.to(q_pre.device)
                # Q still needs RoPE on this layer.
                # K is already RoPEd in the source layer.
                k_dummy = torch.zeros_like(q_pre)
                q_post_, _ = adapter.apply_rope(q_pre, k_dummy, cos, sin)
                q_post = q_post_.transpose(1, 2)
            k_pre_for_capture = None  # do not capture on shared layers
        else:
            k_pre = adapter.apply_k_norm(self, self.k_proj(hidden_states).view(hidden_shape))
            v_post_norm = adapter.apply_v_norm(self, self.v_proj(hidden_states).view(hidden_shape))
            q_post_, k_post_ = adapter.apply_rope(q_pre, k_pre, cos, sin)
            q_post = q_post_.transpose(1, 2)
            key_states = k_post_.transpose(1, 2)
            value_states = v_post_norm.transpose(1, 2)
            k_pre_for_capture = k_pre  # [B, T, Hkv, d]

        q_pre = q_pre.transpose(1, 2)              # [B, Hq, T, d]

        if past_key_values is not None and not is_kv_shared:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if adapter.store_full_length_kv(self):
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
            v_captured = (value_states.transpose(1, 2)                             # [B, T, Hkv, d]
                          [:, pos, :, :].detach())
            capture_bank = ctx.capture_bank
            if capture_bank is not None:
                v_captured = _scale_bank_value_capture(
                    v_captured,
                    mode=getattr(capture_bank, "value_scale_mode", "auto_rms_cap"),
                    has_native_v_norm=adapter.has_native_v_norm(self),
                    target_rms=float(getattr(capture_bank, "value_target_rms", 0.5)),
                    eps=float(getattr(capture_bank, "value_scale_eps", 1e-6)),
                )
            ctx._capture_V[layer_idx] = v_captured

        # --- standard attention ---
        scaling = getattr(self, "scaling", None) or (head_dim ** -0.5)
        # NOTE: sliding-window masking is already baked into ``attention_mask``
        # by HF's eager preparation, so we do not re-apply it here.

        k_repeat = adapter.repeat_kv(key_states, self.num_key_value_groups)
        v_repeat = adapter.repeat_kv(value_states, self.num_key_value_groups)
        scores_orig = torch.matmul(q_post, k_repeat.transpose(2, 3)) * scaling  # [B,Hq,T,Tk]
        if attention_mask is not None:
            scores_orig = scores_orig + attention_mask[..., : scores_orig.size(-1)]

        # --- bank attention (position-agnostic, GQA-aware).
        # KV-shared layers re-use the bank slot from their source layer so the
        # injection is visible in every attention layer, not only non-shared
        # ones (otherwise 20/35 layers in Gemma-4-E2B would skip the bank).
        bank_layer_idx = (
            adapter.kv_shared_index(self) if is_kv_shared else layer_idx
        )
        if bank_layer_idx is None:
            bank_layer_idx = layer_idx
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
            # Stage 14A: optional InfoNCE K-projector (identity-init safe).
            k_proj = getattr(bank, "k_projector", None)
            if k_proj is not None:
                mk = k_proj(bank_layer_idx, mk)
            mk_e = adapter.repeat_kv(mk.unsqueeze(0).transpose(1, 2), self.num_key_value_groups)  # [1, Hq, N, d]
            mv_e = adapter.repeat_kv(mv.unsqueeze(0).transpose(1, 2), self.num_key_value_groups)
            mk_e = mk_e.expand(q_pre.size(0), -1, -1, -1)
            mv_e = mv_e.expand(q_pre.size(0), -1, -1, -1)
            # Stage 15B: optional cosine (L2-normalized) scoring on the bank
            # branch. ``bank_cosine = False`` (default) preserves v3 behavior.
            use_cosine = bool(getattr(bank, "bank_cosine", False))
            if use_cosine:
                q_cos = q_pre / (q_pre.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                k_cos = mk_e / (mk_e.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                scores_bank = torch.matmul(q_cos, k_cos.transpose(2, 3))  # [B,Hq,T,N]
            else:
                scores_bank = torch.matmul(q_pre, mk_e.transpose(2, 3)) * scaling  # [B,Hq,T,N]
            # Stage 14D: optional bank temperature tau (default 1.0 = no-op).
            tau = float(getattr(bank, "bank_temperature", 1.0))
            if tau <= 0.0:
                raise ValueError(
                    f"bank_temperature must be > 0, got {tau!r}. Set bank.bank_temperature "
                    "to a positive float (1.0 = no-op)."
                )
            if tau != 1.0:
                scores_bank = scores_bank / tau
            # Stage 15A: optional bank top-k gating before softmax (structural
            # softmax-dilution fix). bank_topk = 0 (default) preserves v3
            # behavior bit-for-bit.
            topk = int(getattr(bank, "bank_topk", 0) or 0)
            if topk > 0 and topk < scores_bank.size(-1):
                topv, topi = torch.topk(scores_bank, k=topk, dim=-1)
                neg_inf = torch.full_like(scores_bank, float("-inf"))
                scores_bank = neg_inf.scatter(-1, topi, topv)
            # Stage 15C: optional bank-only separate softmax + additive merge.
            # When ``bank_separate_softmax = True`` we run two independent
            # softmaxes over (sequence) and (bank) and combine the readouts as
            #     out = out_orig + beta * out_bank
            # where beta = ``bank_merge_beta`` (default 1.0). This avoids
            # softmax dilution by sequence tokens entirely. Default False keeps
            # v3 bit-equal.
            sep = bool(getattr(bank, "bank_separate_softmax", False))
            if sep:
                w_orig = F.softmax(scores_orig, dim=-1, dtype=torch.float32).to(q_post.dtype)
                w_bank = F.softmax(scores_bank, dim=-1, dtype=torch.float32).to(q_post.dtype)
                beta = float(getattr(bank, "bank_merge_beta", 1.0))
                out_orig = torch.matmul(w_orig, v_repeat)
                out_bank = torch.matmul(w_bank, alpha * mv_e)
                attn_out = (out_orig + beta * out_bank).transpose(1, 2).contiguous()
                weights = w_orig  # for downstream sanity (unused)
            else:
                scores = torch.cat([scores_orig, scores_bank], dim=-1)
                weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_post.dtype)
                T_orig = scores_orig.size(-1)
                # Phase X.1: diagnostic hook — pre-shield bank attention signals.
                # Zero overhead when no recorder is active (_RECORDER is None).
                import deltamemory.diagnostics as _diag_mod  # noqa: PLC0415
                if _diag_mod._RECORDER is not None:
                    _diag_mod._RECORDER.record_bank_attn(layer_idx, weights, T_orig)
                # Stage 16 (v3.2): optional mHC spectral shield.  When
                # ``bank.mhc_shield = True`` bank-column column-sums are
                # capped at ≤ kappa (default 1.0), bounding spectral
                # amplification of the external-KV channel while leaving
                # native sequence columns bit-for-bit unchanged.
                # Default False keeps v3.1 bit-equal.
                if bank.mhc_shield:
                    from deltamemory.memory.mhc_shield import shield_attention_weights

                    bank_n = scores_bank.size(-1)
                    weights = shield_attention_weights(
                        weights, bank_size=bank_n,
                        enabled=True,
                    )
                out_orig = torch.matmul(weights[..., :T_orig], v_repeat)
                out_bank = torch.matmul(weights[..., T_orig:], alpha * mv_e)
                # Stage R (v3.3): optional Dynamic LOPI wrapper.  When
                # ``bank.lopi_cfg.enabled = False`` (default) this is a
                # no-op and the formula stays bit-for-bit identical to v3.2.
                lopi_cfg = getattr(bank, "lopi_cfg", None)
                if lopi_cfg is not None and getattr(lopi_cfg, "enabled", False):
                    from deltamemory.memory.lopi import apply_lopi

                    lopi_state = bank.lopi_state
                    # Phase S (B1 causality fix): on the first layer of each
                    # forward, promote the prior forward's pending norms to
                    # the read-only snapshot that all layers of THIS forward
                    # will read.  Writes from this forward go to pending and
                    # are promoted at the next forward.  Without this swap
                    # the v3.4 docstring's "causality trick" was a no-op for
                    # any layer index > 0.
                    if layer_idx == 0:
                        lopi_state.commit_step()
                    out_bank = apply_lopi(
                        out_bank_native=out_bank,
                        v_ctx_readout=out_orig,
                        q_post=q_post,
                        layer_idx=layer_idx,
                        state=lopi_state,
                        cfg=lopi_cfg,
                    )
                    # Write current-step norm to pending (NOT prev) -- see B1
                    with torch.no_grad():
                        lopi_state.pending_residual_norms[layer_idx] = float(
                            torch.linalg.vector_norm(out_orig, ord=2, dim=-1).mean().item()
                        )
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

    def __init__(self, model, adapter: "ArchAdapter | None" = None):
        from deltamemory.memory.arch_adapter import ArchAdapter, pick_adapter

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

        # Pick or accept an ArchAdapter. The adapter handles family-specific
        # q/k/v norm, RoPE, KV-sharing, and repeat_kv. Without it we cannot
        # be sure the patched forward is bit-equal to the upstream model.
        if adapter is None:
            adapter = pick_adapter(self.attn_modules[0])
        elif not isinstance(adapter, ArchAdapter):
            raise TypeError(f"adapter must be an ArchAdapter, got {type(adapter).__name__}")
        self.adapter: ArchAdapter = adapter

        cfg = getattr(model.config, "text_config", model.config)
        self.num_kv_heads = cfg.num_key_value_heads
        cfg_head_dim = getattr(cfg, "head_dim", None)
        if cfg_head_dim is None:
            cfg_head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.head_dim = int(cfg_head_dim)

        self._orig_forwards: list = [None] * self.num_layers
        self.bank: AttnNativeBank | None = None
        self.alpha: float = 0.0
        self.capture_mode: bool = False
        self.capture_pos: int | None = None
        self.capture_bank: AttnNativeBank | None = None
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
    def capturing(self, capture_pos: int | None = None, bank: AttnNativeBank | None = None):
        self.capture_mode = True
        self.capture_pos = capture_pos
        self.capture_bank = bank
        self._capture_K = [None] * self.num_layers
        self._capture_V = [None] * self.num_layers
        try:
            yield
        finally:
            self.capture_mode = False
            self.capture_bank = None

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

    # Some families (Llama / Mistral) don't store head_dim directly on cfg;
    # derive it from hidden_size / num_attention_heads as fallback.
    cfg_head_dim = getattr(cfg, "head_dim", None)
    if cfg_head_dim is None:
        cfg_head_dim = cfg.hidden_size // cfg.num_attention_heads

    # Discover per-layer head_dim by walking attn modules (Gemma-4 has different
    # head_dim on sliding vs full layers: 256 vs 512).
    patcher_probe = AttnNativePatcher(model)
    head_dims: list[int] = []
    for sa in patcher_probe.attn_modules:
        d = getattr(sa, "head_dim", None) or cfg_head_dim
        head_dims.append(int(d))

    return AttnNativeBank(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=int(cfg_head_dim),
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
    policy: str = "period",
) -> None:
    """Single-shot bank insertion: forward the write_prompt and grab K/V.

    Args:
        policy: capture policy (``"period"`` v2 default, ``"address"``
            Stage 14B, ``"multi"`` Stage 14C). Honors an explicit
            ``capture_pos`` override (pins to ``"period"`` semantics).
    """
    from deltamemory.memory.capture_policy import CaptureSite, resolve_capture_sites

    device = next(patcher.model.parameters()).device
    enc = tokenizer(write_prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)

    if capture_pos is not None:
        sites = [CaptureSite(token_pos=int(capture_pos), role="manual")]
    else:
        sites = resolve_capture_sites(
            policy=policy,
            write_prompt=write_prompt,
            address=address,
            tokenizer=tokenizer,
            attention_mask_row=am[0],
            add_special_tokens=True,
        )

    for site in sites:
        with patcher.patched(), patcher.capturing(capture_pos=site.token_pos, bank=bank), torch.no_grad():
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
            K_per_layer.append(kc[0])
            V_per_layer.append(vc[0])
        site_fact_id = (
            fact_id if len(sites) == 1 else f"{fact_id}@{site.role}"
        )
        bank.append(K_per_layer, V_per_layer, fact_id=site_fact_id, address=address)


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
        out = patcher.model(input_ids=ids, attention_mask=am, use_cache=True)
    last = am.sum(dim=1).item() - 1
    return out.logits[0, last].detach()
