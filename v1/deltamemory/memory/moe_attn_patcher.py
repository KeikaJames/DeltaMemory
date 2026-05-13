"""MoE-aware AttnNativePatcher with per-expert column-cap shield.

This patcher extends :class:`deltamemory.memory.attn_native_bank.AttnNativePatcher`
with three behaviours specific to MoE FFN models (Qwen3-MoE, Mixtral, …):

1. **Router-output capture**: forward hooks on each MoE block stash the
   (expert_indices, routing_weights) returned by the router gate into the
   adapter's per-layer cache.
2. **Per-expert column cap**: the post-softmax bank-attention weights are
   shielded with :func:`deltamemory.memory.mhc_shield.apply_shield_per_expert`
   instead of the global :func:`shield_attention_weights`.
3. **A/B-comparable cap modes**: ``cap_mode`` is one of ``"none"``,
   ``"global"`` (W.1 contract), ``"per_expert"`` (W.5 / X.2 contract).

Red-line invariants (W.5 PREREG / plan.md)
------------------------------------------
* α=0 bit-equality: when ``alpha == 0`` or the bank is empty the dense
  patcher's ``do_inject`` guard short-circuits the entire bank branch
  (including this patcher's shield hook).  We never compute router outputs
  in that path either — see ``do_inject`` short-circuit at
  ``attn_native_bank.py:490-496``.
* No new ``nn.Parameter`` is added.  No LoRA.  Bank weights and router
  weights are read-only.
* The dense path stays unchanged: this module only adds a subclass; the
  existing 197-test baseline is not perturbed.

Implementation note — shield interception
-----------------------------------------
The base ``_make_patched_forward`` calls
``from deltamemory.memory.mhc_shield import shield_attention_weights`` lazily
inside the forward, so monkey-patching ``mhc_shield.shield_attention_weights``
at install time is enough to swap in a per-expert dispatcher.  The dispatcher
reads the *currently active* ``layer_idx`` from a thread-local set by an
attention-module pre-hook, looks up router outputs from the adapter's cache,
and calls :func:`apply_shield_per_expert` (or returns the global cap, or
returns the input unchanged for ``cap_mode='none'``).

This avoids duplicating ~200 lines of attention code while keeping the dense
path bit-equal when the patcher is *not* installed.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

from deltamemory.memory.attn_native_bank import AttnNativePatcher
from deltamemory.memory.arch_moe_adapter import MoeArchSpec, RouterOutputs
from deltamemory.memory.mhc_shield import (
    apply_shield_per_expert,
    shield_attention_weights as _global_shield,
)
import deltamemory.memory.mhc_shield as _mhc_mod

CapMode = Literal["none", "global", "per_expert"]


# Thread-local stack for the layer_idx currently running inside the patched
# forward; pushed by attention pre-hook, popped by attention post-hook.
_TLS = threading.local()


def _push_layer(layer_idx: int) -> None:
    if not hasattr(_TLS, "stack"):
        _TLS.stack = []
    _TLS.stack.append(layer_idx)


def _pop_layer() -> Optional[int]:
    if hasattr(_TLS, "stack") and _TLS.stack:
        return _TLS.stack.pop()
    return None


def _current_layer() -> Optional[int]:
    if hasattr(_TLS, "stack") and _TLS.stack:
        return _TLS.stack[-1]
    return None


def _decode_router_output(raw, num_experts: int) -> Optional[RouterOutputs]:
    """Heuristically unpack an MoE-block router output into RouterOutputs.

    Different MoE architectures expose router outputs differently:

    * Qwen3-MoE ``Qwen3MoeTopKRouter.forward()`` returns
      ``(router_logits, routing_weights, expert_indices)``.
    * Mixtral ``MixtralTopKRouter.forward()`` returns
      ``(router_logits, top_k_weights, top_k_index)``.

    We accept any 2- or 3-tuple where two of the entries are ``(seq_len,
    top_k)`` tensors with matching shape and one entry has integer dtype.
    """
    if not isinstance(raw, (tuple, list)):
        return None
    tensors = [t for t in raw if torch.is_tensor(t)]
    weights = idx = None
    for t in tensors:
        if t.dtype in (torch.long, torch.int64, torch.int32):
            idx = t
        elif t.dim() == 2 and t.dtype.is_floating_point:
            # The shorter-last-dim 2D float tensor is the routing_weights.
            if weights is None or t.size(-1) <= weights.size(-1):
                weights = t
    if idx is None or weights is None:
        return None
    if idx.shape != weights.shape:
        # Some implementations return (B, T, top_k); flatten to (seq, top_k).
        try:
            idx = idx.reshape(-1, idx.size(-1))
            weights = weights.reshape(-1, weights.size(-1))
        except RuntimeError:
            return None
    return RouterOutputs(
        expert_indices=idx.detach(),
        routing_weights=weights.detach().to(torch.float32),
        num_experts=num_experts,
    )


class MoeAttnNativePatcher(AttnNativePatcher):
    """Patcher for FFN-MoE architectures with switchable column-cap mode.

    Args:
        model: a transformers HF model (Qwen3-MoE / Mixtral / synthetic mock).
        adapter: an instance of :class:`MoeArchSpec`.  If ``None``, the base
            class auto-picks a dense adapter; when used on a real MoE model
            the caller must pass an explicit ``adapter`` (the dense registry
            does not include MoE adapters by default).
        cap_mode: one of ``"none"``, ``"global"``, ``"per_expert"``.
            ``"none"`` disables the spectral shield entirely.
            ``"global"`` reuses the W.1 column-sum cap.
            ``"per_expert"`` uses the W.5 per-expert cap.
        kappa: column-sum cap value (same semantics as the dense shield).
    """

    def __init__(
        self,
        model,
        adapter: Optional[MoeArchSpec] = None,
        cap_mode: CapMode = "per_expert",
        kappa: float = 1.0,
    ):
        super().__init__(model, adapter=adapter)
        if cap_mode not in ("none", "global", "per_expert"):
            raise ValueError(
                f"cap_mode must be one of 'none','global','per_expert', got {cap_mode!r}"
            )
        self.cap_mode: CapMode = cap_mode
        self.kappa: float = float(kappa)

        # Hooks installed at install() / removed at remove().
        self._moe_block_hooks: list = []
        self._attn_pre_hooks: list = []
        self._attn_post_hooks: list = []
        self._orig_shield_fn = None

    # ------------------------------------------------------------------
    # Hook factories
    # ------------------------------------------------------------------

    def _make_router_hook(self, layer_idx: int):
        adapter = self.adapter
        num_experts = (
            adapter.get_num_experts(layer_idx)
            if isinstance(adapter, MoeArchSpec)
            else 0
        )

        def _hook(module, inputs, output):
            ro = _decode_router_output(output, num_experts)
            if ro is not None and isinstance(adapter, MoeArchSpec):
                adapter.set_router_outputs(layer_idx, ro)

        return _hook

    def _make_attn_pre_hook(self, layer_idx: int):
        def _pre_hook(module, args, kwargs):
            _push_layer(layer_idx)
            return None

        return _pre_hook

    def _make_attn_post_hook(self, layer_idx: int):
        def _post_hook(module, args, output):
            _pop_layer()
            return None

        return _post_hook

    # ------------------------------------------------------------------
    # Dispatcher: replaces shield_attention_weights at install time
    # ------------------------------------------------------------------

    def _dispatch_shield(
        self,
        weights: torch.Tensor,
        bank_size: int,
        enabled: bool,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """Replacement for ``mhc_shield.shield_attention_weights``.

        Honors ``self.cap_mode``.  When ``per_expert`` is selected and
        router outputs are available for the current layer, dispatches to
        :func:`apply_shield_per_expert`; otherwise falls back to the global
        cap (so ``per_expert`` on a non-MoE layer degrades to global, not to
        a runtime error).
        """
        # cap_mode == "none": the dense forward only calls this when
        # ``bank.mhc_shield`` is True; we honor cap_mode by returning input
        # unchanged.
        if self.cap_mode == "none" or not enabled or bank_size <= 0:
            return weights

        # Use the patcher's kappa override if the caller passed the default.
        eff_kappa = self.kappa if kappa == 1.0 else float(kappa)

        if self.cap_mode == "global":
            return _global_shield(weights, bank_size=bank_size, enabled=True, kappa=eff_kappa)

        # cap_mode == "per_expert"
        layer_idx = _current_layer()
        adapter = self.adapter
        ro: Optional[RouterOutputs] = None
        if layer_idx is not None and isinstance(adapter, MoeArchSpec):
            ro = adapter.get_router_outputs(layer_idx)
        if ro is None:
            # No MoE block at this layer (or capture failed): fall back to
            # the global cap so the shield still bounds spectral norm.
            return _global_shield(weights, bank_size=bank_size, enabled=True, kappa=eff_kappa)

        T_orig = weights.size(-1) - bank_size
        # Reshape weights to a 2-D view (q, T+N) per the
        # apply_shield_per_expert contract.  Original layout is
        # (B, Hq, T, T+N) — we collapse (B, Hq, T) into a single q axis.
        orig_shape = weights.shape
        flat = weights.reshape(-1, orig_shape[-1])

        # Build per-expert gate dict from the captured top-k tensors.
        # Each gate vector must have length == flat.size(0); we tile the
        # router gates (which are per *token*, length B*T) across heads.
        seq_len = ro.expert_indices.size(0)
        # B*Hq*T = flat.size(0); tile by Hq-equivalent factor:
        rep = flat.size(0) // seq_len if seq_len > 0 else 0
        if rep == 0 or seq_len * rep != flat.size(0):
            # Shape mismatch (e.g. test feeds 2D weights directly): use
            # router shape unchanged.
            rep = 1

        expert_gates: Dict[int, torch.Tensor] = {}
        device = flat.device
        dtype = torch.float32
        for e in range(ro.num_experts):
            mask = ro.expert_indices.eq(e)              # (seq_len, top_k)
            if not bool(mask.any()):
                continue
            gate = (ro.routing_weights * mask.to(ro.routing_weights.dtype)).sum(dim=-1)
            gate = gate.to(device=device, dtype=dtype)  # (seq_len,)
            if rep != 1:
                # Repeat each token's gate across heads/queries for the
                # collapsed axis.  Pattern matches (B, Hq, T) -> flatten.
                gate = gate.repeat_interleave(rep, dim=0) if rep * seq_len == flat.size(0) else gate
            expert_gates[e] = gate

        if not expert_gates:
            return weights

        shielded = apply_shield_per_expert(
            flat, T_orig=T_orig, kappa=eff_kappa, expert_gates=expert_gates
        )
        return shielded.reshape(orig_shape)

    # ------------------------------------------------------------------
    # Install / remove
    # ------------------------------------------------------------------

    def install(self) -> None:
        super().install()
        # Clear router cache from any prior forward.
        if isinstance(self.adapter, MoeArchSpec):
            self.adapter.clear_router_cache()

        # Find decoder layers (re-walk the path used by the base class).
        layers = self._find_decoder_layers()
        adapter = self.adapter

        # 1) Router hooks on each MoE block.
        if layers is not None and isinstance(adapter, MoeArchSpec):
            for i, layer in enumerate(layers):
                block = adapter.get_moe_block(layer)
                gate = getattr(block, "gate", None) if block is not None else None
                if gate is not None:
                    h = gate.register_forward_hook(self._make_router_hook(i))
                    self._moe_block_hooks.append(h)

        # 2) Pre/post hooks on attention modules to maintain layer_idx TLS.
        for i, m in enumerate(self.attn_modules):
            pre = m.register_forward_pre_hook(self._make_attn_pre_hook(i), with_kwargs=True)
            post = m.register_forward_hook(self._make_attn_post_hook(i))
            self._attn_pre_hooks.append(pre)
            self._attn_post_hooks.append(post)

        # 3) Monkey-patch the shield function so the base patched forward
        #    dispatches to our cap_mode-aware version.
        self._orig_shield_fn = _mhc_mod.shield_attention_weights
        _mhc_mod.shield_attention_weights = self._dispatch_shield

    def remove(self) -> None:
        # 3) restore shield first
        if self._orig_shield_fn is not None:
            _mhc_mod.shield_attention_weights = self._orig_shield_fn
            self._orig_shield_fn = None
        # 2) attn hooks
        for h in self._attn_pre_hooks:
            h.remove()
        for h in self._attn_post_hooks:
            h.remove()
        self._attn_pre_hooks.clear()
        self._attn_post_hooks.clear()
        # Drain TLS stack
        if hasattr(_TLS, "stack"):
            _TLS.stack.clear()
        # 1) router hooks
        for h in self._moe_block_hooks:
            h.remove()
        self._moe_block_hooks.clear()
        # base
        super().remove()

    def _find_decoder_layers(self):
        """Re-walk model attribute paths to locate the decoder layer list.

        Mirrors the logic used by :class:`AttnNativePatcher.__init__` so we
        get the *layer* objects (whose ``.mlp`` / ``.block_sparse_moe`` is
        the MoE block), not just the attention modules.
        """
        for path in (
            "model.model.language_model.layers",
            "model.model.layers",
            "model.language_model.model.layers",
            "model.language_model.layers",
            "language_model.layers",
            "model.layers",
        ):
            obj = self.model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok and hasattr(obj, "__len__") and len(obj) > 0 and hasattr(obj[0], "self_attn"):
                return obj
        return None


__all__ = ["MoeAttnNativePatcher", "CapMode"]
