"""MoE-aware ArchAdapter bridge for Mneme's bank patcher (Phase W.5).

This module extends the dense :class:`deltamemory.memory.arch_adapter.ArchAdapter`
interface with the two extra hooks the per-expert column-cap shield needs:

* :meth:`MoeArchSpec.get_router_outputs` — pull the top-k *(expert_indices,
  routing_weights)* tensors that the MoE block produced for the current
  forward pass at a given decoder layer.
* :meth:`MoeArchSpec.get_active_experts` — top-k value (number of experts
  selected per token).

Theory: ``docs/theory/mhc_moe.md`` (U.2).  Plan: W.5 / W-T2 in
``/Users/gabiri/.copilot/session-state/.../plan.md``.

Red-line invariants
-------------------
1. **No new nn.Parameters and no LoRA**: this adapter is read-only — it never
   modifies weights.  The bank K/V is still the only "external" tensor, and
   it stays parameter-free.
2. **α=0 bit-equality is preserved**: the patcher short-circuits the bank
   branch entirely when ``alpha == 0`` or the bank is empty (see
   :class:`deltamemory.memory.moe_attn_patcher.MoeAttnNativePatcher`).  The
   adapter is never consulted in that path.
3. **Dense path untouched**: this module only adds new classes; the existing
   dense ``ArchAdapter`` registry is not mutated.

Concrete adapters
-----------------
* :class:`Qwen3MoeAdapter` — for ``Qwen3MoeAttention`` (Qwen3-MoE-A3B,
  Qwen3.5-35B-A3B-Base).  Inherits the dense Qwen3 RoPE / q-norm / k-norm
  conventions from :class:`deltamemory.memory.arch_adapter.Qwen3Adapter`.
* :class:`MockMoeAdapter` — synthetic adapter with no real model behind it.
  Used by unit tests when no MoE model is locally cached.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from deltamemory.memory.arch_adapter import (
    ArchAdapter,
    LlamaAdapter,
    Qwen3Adapter,
)


@dataclass
class RouterOutputs:
    """Captured top-k router outputs for one decoder layer / one forward pass.

    Attributes:
        expert_indices: ``(seq_len, top_k)`` long tensor of expert ids selected
            per token.  ``seq_len = B * T`` flattened (matches the convention
            in transformers' MoE blocks).
        routing_weights: ``(seq_len, top_k)`` float tensor of (post-softmax,
            optionally post-norm) gate weights for each selected expert.
        num_experts: total number of experts in the layer's pool (``E``).
    """

    expert_indices: torch.Tensor
    routing_weights: torch.Tensor
    num_experts: int


class MoeArchSpec(ArchAdapter):
    """ArchAdapter extension that exposes router outputs for MoE models.

    Subclasses must implement :meth:`get_router_outputs` and
    :meth:`get_active_experts`.  They MAY also override the dense
    :class:`ArchAdapter` methods (RoPE, q/k/v-norm, repeat_kv) when the
    underlying attention class needs family-specific handling.

    This base class provides a default in-memory router cache: the
    :class:`MoeAttnNativePatcher` writes captured router outputs into
    ``self._router_cache[layer_idx]`` via a forward hook on each MoE block,
    and ``get_router_outputs`` simply reads from that cache.  Subclasses can
    override this when an architecture exposes router outputs differently
    (e.g. ``output_router_logits=True`` returning logits instead of top-k).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._router_cache: Dict[int, RouterOutputs] = {}

    # ------------------------------------------------------------------
    # MoE-specific hooks
    # ------------------------------------------------------------------

    def get_router_outputs(
        self,
        layer_idx: int,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Optional[RouterOutputs]:
        """Return router outputs for ``layer_idx`` from the most recent forward.

        Args:
            layer_idx: 0-based decoder layer index.
            hidden_states: optional input to the MoE block (unused by the
                default cache-based implementation; subclasses that need to
                recompute gates can use it).

        Returns:
            :class:`RouterOutputs` or ``None`` if the layer is not an MoE
            layer or no forward has captured its router output yet.
        """
        return self._router_cache.get(int(layer_idx))

    def set_router_outputs(self, layer_idx: int, outputs: RouterOutputs) -> None:
        """Cache router outputs for ``layer_idx`` (called by the patcher)."""
        self._router_cache[int(layer_idx)] = outputs

    def clear_router_cache(self) -> None:
        """Wipe all cached router outputs (called between forwards)."""
        self._router_cache.clear()

    def get_active_experts(self, layer_idx: int = 0) -> int:
        """Top-k value: number of experts selected per token.

        Default is 2 (Mixtral); subclasses override.
        """
        return 2

    def get_num_experts(self, layer_idx: int = 0) -> int:
        """Total experts in the layer's pool. Subclasses override."""
        return 8

    def get_moe_block(self, decoder_layer: nn.Module) -> Optional[nn.Module]:
        """Return the MoE block within ``decoder_layer`` (the gate's parent).

        Used by :class:`MoeAttnNativePatcher` to install router hooks.
        Default tries common attribute names: ``mlp`` (Qwen3MoE),
        ``block_sparse_moe`` (Mixtral).  Returns None for dense layers.
        """
        for attr in ("mlp", "block_sparse_moe", "feed_forward"):
            block = getattr(decoder_layer, attr, None)
            if block is None:
                continue
            # Heuristic: an MoE block has a ``gate`` submodule and at least
            # one of ``experts`` / ``num_experts`` attribute.
            if hasattr(block, "gate") and (
                hasattr(block, "experts") or hasattr(block, "num_experts")
            ):
                return block
        return None


# ---------------------------------------------------------------------------
# Qwen3-MoE
# ---------------------------------------------------------------------------


class Qwen3MoeAdapter(MoeArchSpec, Qwen3Adapter):
    """Adapter for ``Qwen3MoeAttention`` (Qwen3-MoE-A3B, Qwen3.5-35B-A3B).

    Reuses :class:`Qwen3Adapter` for the dense q/k-norm + RoPE conventions
    (Qwen3-MoE keeps Qwen3's attention structure; only the MLP is sparse).

    Router output convention (transformers ≥ 5.7)
    ---------------------------------------------
    The ``Qwen3MoeSparseMoeBlock`` calls its ``gate`` submodule which returns
    ``(router_logits, routing_weights, expert_indices)``.  When
    ``norm_topk_prob`` is True the routing weights are L1-normalised across
    the top-k axis; otherwise they are raw softmax probabilities.
    """

    def __init__(self, num_experts: int = 128, top_k: int = 8):
        Qwen3Adapter.__init__(self)
        # Re-init the router cache (multiple-inheritance dance).
        self._router_cache: Dict[int, RouterOutputs] = {}
        self._num_experts = int(num_experts)
        self._top_k = int(top_k)
        self.name = "qwen3_moe"

    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        n = type(attn_module).__name__
        # The MoE attention class is still ``Qwen3MoeAttention`` in HF; the
        # dense Qwen3Adapter would also match this name (it greps "Qwen3"),
        # so the MoE adapter must be registered with HIGHER priority.
        return "Qwen3Moe" in n or "Qwen3MoE" in n

    def get_active_experts(self, layer_idx: int = 0) -> int:
        return self._top_k

    def get_num_experts(self, layer_idx: int = 0) -> int:
        return self._num_experts


# ---------------------------------------------------------------------------
# Mock MoE adapter (for tests when no real MoE model is cached locally)
# ---------------------------------------------------------------------------


class MockMoeAdapter(MoeArchSpec, LlamaAdapter):
    """Synthetic MoE adapter — exercises per-expert cap math without weights.

    Use this in unit tests and in CI environments where no real MoE model is
    cached.  It piggybacks on :class:`LlamaAdapter` for the dense attention
    primitives (since the synthetic models we instantiate in tests use a
    Llama-style attention module), and overrides only the MoE-specific
    hooks to feed pre-computed router outputs from a fixture.

    What this validates
    -------------------
    * The per-expert cap formula matches ``apply_shield_per_expert`` outputs.
    * The dispatcher branches between cap_mode={none, global, per_expert}
      correctly.
    * α=0 bit-equality holds regardless of cap_mode.

    What this does NOT validate (deferred to GB10 / a real cached MoE)
    -----------------------------------------------------------------
    * End-to-end drift on a real Qwen3-MoE / Mixtral checkpoint.
    * That FFN-router-gate proxy timing (router fires AFTER attention in the
      same decoder block) is acceptable on real text — see W-T2.2 in plan.
    """

    def __init__(self, num_experts: int = 4, top_k: int = 2):
        LlamaAdapter.__init__(self)
        self._router_cache: Dict[int, RouterOutputs] = {}
        self._num_experts = int(num_experts)
        self._top_k = int(top_k)
        self.name = "mock_moe"

    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        # Mock adapter is never auto-picked by the registry; tests construct
        # it explicitly.
        return False

    def get_active_experts(self, layer_idx: int = 0) -> int:
        return self._top_k

    def get_num_experts(self, layer_idx: int = 0) -> int:
        return self._num_experts

    def get_moe_block(self, decoder_layer: nn.Module) -> Optional[nn.Module]:
        # Mock layers don't have an MoE block; the patcher is responsible
        # for feeding router outputs via :meth:`set_router_outputs`.
        return None


def make_moe_adapter_from_dense(
    dense_adapter: ArchAdapter,
    num_experts: int = 4,
    top_k: int = 2,
) -> MoeArchSpec:
    """Wrap a *dense* ArchAdapter as an :class:`MoeArchSpec`-conformant adapter.

    The dynamic subclass inherits all dense behaviour (RoPE, q/k/v-norm,
    repeat_kv, KV-sharing) from ``dense_adapter`` so bit-equality of the
    base patcher is preserved.  MoE-specific hooks (router cache, top-k,
    get_moe_block) come from :class:`MoeArchSpec`.

    This is the recommended way to plug a synthetic / mock MoE adapter onto
    a real *dense* model for unit tests when no MoE checkpoint is cached.
    """
    base_cls = type(dense_adapter)

    class _DynamicMoeAdapter(MoeArchSpec, base_cls):  # type: ignore[misc, valid-type]
        def __init__(self):
            base_cls.__init__(self)
            self._router_cache = {}
            self._num_experts = int(num_experts)
            self._top_k = int(top_k)
            self.name = f"mock_moe_over_{getattr(dense_adapter, 'name', 'base')}"

        @classmethod
        def matches(cls, attn_module):  # never auto-picked
            return False

        def get_active_experts(self, layer_idx: int = 0) -> int:
            return self._top_k

        def get_num_experts(self, layer_idx: int = 0) -> int:
            return self._num_experts

        def get_moe_block(self, decoder_layer):
            return None

    return _DynamicMoeAdapter()


__all__ = [
    "RouterOutputs",
    "MoeArchSpec",
    "Qwen3MoeAdapter",
    "MockMoeAdapter",
    "make_moe_adapter_from_dense",
]
