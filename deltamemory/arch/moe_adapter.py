"""MoE architecture adapters for the Mneme per-expert shield (Phase X.2).

Architectural scope — FFN-MoE vs MoE-Attention
-----------------------------------------------
All currently supported models (Mixtral 8×7B, Qwen3-MoE) use **FFN-MoE**:
the MoE routing selects experts for the feed-forward sublayer while the
*attention sublayer remains dense*.  The bank-injection in mHC happens at
the attention sublayer.  Therefore:

  * In FFN-MoE architectures the per-expert cap is an **approximation**:
    we use the FFN router gate g_e(token) as a proxy for how much of
    each token's attention output "belongs to" expert e, then apply an
    independent column cap per expert in that bucketed view.

  * In a true MoE-Attention architecture (expert-specific W_v; rare), the
    per-expert cap would be exact.  No such architecture is currently
    wired in this module.

The ``MoEArchAdapter`` interface is designed to be correct for both cases.
The ``mhc_moe.md`` theory doc clarifies the distinction in full.

W.5 re-visit note
-----------------
Before running the W.5 MoE sweep, verify whether the FFN router gates
at decoder layer L are the right proxy for the attention cap at the same
layer L (they are computed AFTER attention in the forward pass, so a
one-layer lag exists).  This may require passing the gates from layer L-1
into the attention cap of layer L.  This infrastructure phase intentionally
defers that question.

Shared-expert note
------------------
The HuggingFace implementation of Qwen3-MoE (transformers ≥ 5.7.0) does NOT
include a dedicated shared-expert sub-module.  All ``num_local_experts``
experts are routable; none is "always active".  The ``is_shared_expert``
method always returns ``False`` for Qwen3MoEAdapter.  If a future Qwen
variant adds shared experts (as in Qwen2-MoE / DeepSeek-V3), subclass this
adapter and override ``is_shared_expert``.

Code path for real forward passes
----------------------------------
Neither ``Qwen3MoEAdapter`` nor ``MixtralAdapter`` require the LLM weights
to be loaded; both can be instantiated with a ``config`` dict or a
HuggingFace config object.  Live gate extraction (``get_router_gates``)
requires the matching HuggingFace model to be loaded and patched — that
wiring is left for W.5.

Usage
-----
::

    adapter = Qwen3MoEAdapter(config)
    # During a patched forward: (top_k_weights, top_k_indices) returned by
    # the gate module are passed in:
    gates = adapter.decode_gate_output(
        routing_weights=top_k_weights,   # (seq_len, top_k)
        expert_indices=top_k_indices,    # (seq_len, top_k) int
    )
    # gates: Dict[expert_id (int), gate_tensor (seq_len,)]
    # Missing experts get zero gate.
"""
from __future__ import annotations

import abc
from typing import Dict, List

import torch


class MoEArchAdapter(abc.ABC):
    """Abstract base class for MoE architecture adapters.

    An adapter exposes three hooks consumed by
    ``mhc_shield.apply_shield_per_expert``:

    * :meth:`get_router_gates` — live gate lookup during a patched forward.
    * :meth:`get_expert_pool`  — static list of expert ids for a layer.
    * :meth:`is_shared_expert` — whether an expert id is a "shared" (always-
      active) expert.  Shared experts receive gate ``1.0`` for every token.

    Sub-classes must implement all three abstract methods.

    Architectural scope note
    ~~~~~~~~~~~~~~~~~~~~~~~~
    This adapter operates on FFN-MoE router gates, NOT on attention-layer
    MoE routing.  See module docstring for the full discussion.
    """

    @abc.abstractmethod
    def get_router_gates(
        self,
        layer_idx: int,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Return per-expert gate tensors for a single decoder layer.

        This method converts the raw router output (top-k weights and indices)
        into a dense Dict mapping each expert id to its gate vector over all
        tokens in the current batch.

        Args:
            layer_idx: Index of the transformer decoder layer (0-based).
            routing_weights: Float tensor of shape ``(seq_len, top_k)`` with
                softmax-normalised routing weights for the top-k experts.
            expert_indices: Long tensor of shape ``(seq_len, top_k)`` with
                the indices of the selected top-k experts per token.

        Returns:
            Mapping from expert id (int) to a ``(seq_len,)`` float tensor of
            gate values.  Experts not selected by any token have gate ``0``.
            For shared experts :meth:`is_shared_expert` is True and their
            gate tensor is all-ones (gate = 1.0 for every token).
        """

    @abc.abstractmethod
    def get_expert_pool(self, layer_idx: int) -> List[int]:
        """Return the list of valid expert ids for the given layer.

        For most MoE models all layers have the same expert pool
        ``[0, 1, ..., num_local_experts - 1]``, but some models (e.g. those
        with ``mlp_only_layers``) have dense MLP layers with no experts.

        Returns:
            Sorted list of expert ids; empty list if the layer has no MoE.
        """

    @abc.abstractmethod
    def is_shared_expert(self, expert_id: int) -> bool:
        """Return True if ``expert_id`` is a shared (always-active) expert.

        Shared experts always receive gate value ``1.0`` regardless of the
        router output.  They are present in e.g. DeepSeek-V3 and Qwen2-MoE.
        Qwen3-MoE (transformers ≥ 5.7) does NOT have shared experts.

        Args:
            expert_id: Integer expert identifier.

        Returns:
            ``True`` if this expert is always active, ``False`` otherwise.
        """

    # ------------------------------------------------------------------
    # Concrete helper shared by all sub-classes
    # ------------------------------------------------------------------

    def decode_gate_output(
        self,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        num_experts: int | None = None,
        layer_idx: int = 0,
    ) -> Dict[int, torch.Tensor]:
        """Convenience method: calls ``get_router_gates`` with validated args.

        This is the primary entry point for callers that have already captured
        the router output tensors during a hooked forward pass.
        """
        return self.get_router_gates(
            layer_idx=layer_idx,
            routing_weights=routing_weights,
            expert_indices=expert_indices,
        )


# ---------------------------------------------------------------------------
# Qwen3-MoE adapter
# ---------------------------------------------------------------------------


class Qwen3MoEAdapter(MoEArchAdapter):
    """Adapter for the Qwen/Qwen3-30B-A3B and Qwen3-MoE-A1.5B model families.

    Architecture summary
    ~~~~~~~~~~~~~~~~~~~~
    * Dense attention (shared K/V across experts).
    * FFN experts: ``num_local_experts`` routable experts, top-``num_experts_per_tok``
      selected per token.
    * No dedicated shared expert in the transformers ≥ 5.7 implementation.
      ``is_shared_expert`` always returns ``False``.
    * Router: ``Qwen3MoeTopKRouter`` — linear gate, softmax + top-k selection,
      optional top-k normalisation (``norm_topk_prob``).

    Gate access path (transformers 5.7.0)::

        layer = model.model.layers[layer_idx]
        moe_block = layer.mlp          # Qwen3MoeSparseMoeBlock
        gate_module = moe_block.gate   # Qwen3MoeTopKRouter

        # During patched forward, intercept output of gate_module.forward():
        # (router_logits, routing_weights, expert_indices)
        # routing_weights: (seq_len, top_k)   — normalised gate weights
        # expert_indices:  (seq_len, top_k)   — int indices of top-k experts

    For the ``mlp_only_layers`` config field: layers listed there use a dense
    MLP instead of a sparse MoE block and have no experts.

    Shared-expert note: Qwen3-MoE does NOT have dedicated shared experts.
    See module docstring.
    """

    def __init__(self, config: object) -> None:
        """Initialise from a HuggingFace ``Qwen3MoeConfig`` or dict.

        Args:
            config: Either a ``transformers.Qwen3MoeConfig`` instance or a
                dict with the following keys:
                ``num_local_experts``, ``num_experts_per_tok``,
                ``num_hidden_layers``, ``decoder_sparse_step``,
                ``mlp_only_layers`` (optional, default ``[]``),
                ``norm_topk_prob`` (optional, default ``False``).
        """
        if isinstance(config, dict):
            self._num_experts: int = int(config["num_local_experts"])
            self._top_k: int = int(config["num_experts_per_tok"])
            self._num_layers: int = int(config["num_hidden_layers"])
            self._sparse_step: int = int(config.get("decoder_sparse_step", 1))
            self._mlp_only: set[int] = set(config.get("mlp_only_layers", []))
            self._norm_topk: bool = bool(config.get("norm_topk_prob", False))
        else:
            self._num_experts = int(getattr(config, "num_local_experts", 64))
            self._top_k = int(getattr(config, "num_experts_per_tok", 8))
            self._num_layers = int(getattr(config, "num_hidden_layers", 28))
            self._sparse_step = int(getattr(config, "decoder_sparse_step", 1))
            self._mlp_only = set(getattr(config, "mlp_only_layers", []) or [])
            self._norm_topk = bool(getattr(config, "norm_topk_prob", False))

    def _is_moe_layer(self, layer_idx: int) -> bool:
        """Return True if layer_idx uses a sparse MoE block."""
        if layer_idx in self._mlp_only:
            return False
        return (layer_idx + 1) % self._sparse_step == 0

    def get_expert_pool(self, layer_idx: int) -> List[int]:
        """Return the expert ids for *layer_idx*.

        Returns an empty list for dense (mlp_only) layers.
        """
        if not self._is_moe_layer(layer_idx):
            return []
        return list(range(self._num_experts))

    def is_shared_expert(self, expert_id: int) -> bool:  # noqa: ARG002
        """Qwen3-MoE has no dedicated shared experts; always False.

        If a future Qwen variant (e.g. Qwen4-MoE) adds shared experts,
        subclass this adapter and override this method.
        """
        return False

    def get_router_gates(
        self,
        layer_idx: int,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Build the per-expert gate map from top-k router output tensors.

        Args:
            layer_idx: Decoder layer index (0-based).
            routing_weights: ``(seq_len, top_k)`` normalised gate weights from
                ``Qwen3MoeTopKRouter.forward()``.
            expert_indices: ``(seq_len, top_k)`` integer expert indices.

        Returns:
            Dict mapping each expert_id to a ``(seq_len,)`` gate tensor.
            Experts not selected by any token have gate ``0``.

        Notes
        -----
        For shared experts ``is_shared_expert`` would return True and they
        would receive gate ``1.0``.  In Qwen3-MoE this never fires.
        """
        if not self._is_moe_layer(layer_idx):
            return {}

        seq_len, top_k = routing_weights.shape
        device = routing_weights.device
        dtype = routing_weights.dtype

        gates: Dict[int, torch.Tensor] = {}
        for e in range(self._num_experts):
            if self.is_shared_expert(e):
                gates[e] = torch.ones(seq_len, dtype=dtype, device=device)
                continue
            # mask: (seq_len, top_k) boolean — is expert e selected?
            mask = expert_indices.eq(e)           # (seq_len, top_k)
            if not mask.any():
                continue
            # gate for token i = sum over top-k positions where expert e is selected
            gate = (routing_weights * mask.to(dtype)).sum(dim=-1)  # (seq_len,)
            gates[e] = gate
        return gates


# ---------------------------------------------------------------------------
# Mixtral adapter
# ---------------------------------------------------------------------------


class MixtralAdapter(MoEArchAdapter):
    """Adapter for mistralai/Mixtral-8x7B-v0.1.

    Architecture summary
    ~~~~~~~~~~~~~~~~~~~~
    * Dense attention (no expert-specific K/V projections in the HuggingFace
      transformers implementation).
    * FFN experts: 8 routable experts, top-2 selected per token (``top_k=2``).
    * No shared experts; ``is_shared_expert`` always returns ``False``.
    * Router: ``MixtralTopKRouter`` — similar to Qwen3MoeTopKRouter.

    Gate access path (transformers 5.7.0)::

        layer = model.model.layers[layer_idx]
        moe_block = layer.block_sparse_moe   # MixtralSparseMoeBlock
        gate_module = moe_block.gate         # MixtralTopKRouter

        # During patched forward, intercept output of gate_module.forward():
        # (router_logits, top_k_weights, top_k_index)
        # top_k_weights: (seq_len, top_k=2)
        # top_k_index:   (seq_len, top_k=2) int

    All decoder layers in Mixtral use sparse MoE; there are no dense MLP
    layers.  ``get_expert_pool`` returns ``[0..7]`` for every layer.
    """

    def __init__(self, config: object) -> None:
        """Initialise from a HuggingFace ``MixtralConfig`` or dict.

        Args:
            config: ``transformers.MixtralConfig`` or dict with keys
                ``num_local_experts`` (default 8), ``num_experts_per_tok``
                (default 2), ``num_hidden_layers`` (default 32).
        """
        if isinstance(config, dict):
            self._num_experts: int = int(config.get("num_local_experts", 8))
            self._top_k: int = int(config.get("num_experts_per_tok", 2))
            self._num_layers: int = int(config.get("num_hidden_layers", 32))
        else:
            self._num_experts = int(getattr(config, "num_local_experts", 8))
            self._top_k = int(getattr(config, "num_experts_per_tok", 2))
            self._num_layers = int(getattr(config, "num_hidden_layers", 32))

    def get_expert_pool(self, layer_idx: int) -> List[int]:
        """All Mixtral layers are sparse MoE; always returns ``[0..E-1]``."""
        if layer_idx < 0 or layer_idx >= self._num_layers:
            return []
        return list(range(self._num_experts))

    def is_shared_expert(self, expert_id: int) -> bool:  # noqa: ARG002
        """Mixtral has no shared experts; always returns ``False``."""
        return False

    def get_router_gates(
        self,
        layer_idx: int,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Dict[int, torch.Tensor]:
        """Build the per-expert gate map from Mixtral top-k router output.

        Args:
            layer_idx: Decoder layer index.
            routing_weights: ``(seq_len, top_k=2)`` normalised gate weights.
            expert_indices: ``(seq_len, top_k=2)`` integer expert indices.

        Returns:
            Dict mapping expert_id to ``(seq_len,)`` gate tensor.
            Experts not selected receive gate ``0`` (not included in dict).
        """
        if not self.get_expert_pool(layer_idx):
            return {}

        seq_len, _ = routing_weights.shape
        dtype = routing_weights.dtype
        device = routing_weights.device

        gates: Dict[int, torch.Tensor] = {}
        for e in range(self._num_experts):
            mask = expert_indices.eq(e)
            if not mask.any():
                continue
            gate = (routing_weights * mask.to(dtype)).sum(dim=-1)
            gates[e] = gate
        return gates
