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
BankKeyMode = Literal["pre_rope", "post_rope"]


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
    # Gemma-4 has heterogeneous num_kv_heads across layers (e.g. 16 on most
    # layers, 4 on use_alternative_attention layers). num_kv_heads_per_layer
    # tracks per-layer counts; falls back to uniform num_kv_heads if unset.
    num_kv_heads_per_layer: list[int] = field(default_factory=list)
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
    # Per-bank kappa for the mHC spectral shield.  Overrides the default
    # kappa=1.0 so Exp8 can sweep kappa without rebuilding the bank.
    mhc_kappa: float = 1.0
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
    # Paper-ablation knob: default production path stores and scores bank K in
    # pre-RoPE coordinates. ``post_rope`` is intentionally opt-in for the
    # AttnNativeBank paper's position-invariance ablation.
    bank_key_mode: BankKeyMode = "pre_rope"

    def __post_init__(self) -> None:
        if not self.head_dims:
            self.head_dims = [self.head_dim] * self.num_layers
        if not self.num_kv_heads_per_layer:
            self.num_kv_heads_per_layer = [self.num_kv_heads] * self.num_layers
        if not self.M_K:
            for layer in range(self.num_layers):
                d = self.head_dims[layer]
                h = self.num_kv_heads_per_layer[layer]
                self.M_K.append(torch.empty(0, h, d,
                                            device=self.device, dtype=self.dtype))
                self.M_V.append(torch.empty(0, h, d,
                                            device=self.device, dtype=self.dtype))
        # Lazy import to keep top-of-file clean and avoid circulars.
        if self.lopi_cfg is None:
            from deltamemory.memory.lopi import LOPIConfig
            self.lopi_cfg = LOPIConfig()  # enabled=False by default
        if self.lopi_state is None:
            from deltamemory.memory.lopi import LOPIState
            self.lopi_state = LOPIState(num_layers=self.num_layers)
        # Phase X.7 — bank lifecycle (default = unbounded, bit-equal to v0.4).
        # See experiments/X7_forget_merge/PREREG.md for the locked design.
        if not hasattr(self, "bank_capacity"):
            self.bank_capacity: int = 0
        if not hasattr(self, "bank_evict_policy"):
            self.bank_evict_policy: str = "lru"
        self._x7_global_step: int = 0
        self._x7_write_step: list[int] = []
        self._x7_last_access: list[int] = []
        self._x7_access_count: list[int] = []
        # Track-M smart memory flags. Defaults are OFF to preserve α=0 and
        # no-bank bit-equality. These are runtime attributes, not Parameters.
        for name, value in (
            ("enable_compression", False),
            ("enable_decay", False),
            ("enable_importance", False),
            ("enable_tiering", False),
            ("compression_threshold", 0),
            ("compression_target_size", 0),
            ("compression_min_similarity", 0.90),
            ("decay_half_life", 1000),
            ("decay_erase_threshold", 1e-3),
        ):
            if not hasattr(self, name):
                setattr(self, name, value)
        self.merge_counts: list[float] = []
        self.importance_scores: list[float] = []
        self.original_v_norm: list[float] = []

    @property
    def empty(self) -> bool:
        return self.size == 0

    @property
    def size(self) -> int:
        return self.M_K[0].size(0) if self.M_K else 0

    def clear(self) -> None:
        for layer in range(self.num_layers):
            d = self.head_dims[layer]
            h = self.num_kv_heads_per_layer[layer]
            self.M_K[layer] = torch.empty(0, h, d,
                                          device=self.device, dtype=self.dtype)
            self.M_V[layer] = torch.empty(0, h, d,
                                          device=self.device, dtype=self.dtype)
        self.fact_ids.clear()
        self.address_strs.clear()
        self._x7_write_step.clear()
        self._x7_last_access.clear()
        self._x7_access_count.clear()
        self.merge_counts.clear()
        self.importance_scores.clear()
        self.original_v_norm.clear()
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
        if bool(getattr(self, "enable_importance", False)):
            from deltamemory.memory.bank_importance import compute_novelty

            novelty = compute_novelty(per_layer_K[0], self.state_dict())
        else:
            novelty = 1.0
        original_norm = float(sum(
            torch.linalg.vector_norm(v.detach().float(), ord=2).item()
            for v in per_layer_V
        ))
        for layer in range(self.num_layers):
            d = self.head_dims[layer]
            h = self.num_kv_heads_per_layer[layer]
            expected = (h, d)
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
        self._x7_global_step += 1
        self._x7_write_step.append(self._x7_global_step)
        self._x7_last_access.append(self._x7_global_step)
        self._x7_access_count.append(0)
        self.merge_counts.append(1.0)
        self.importance_scores.append(float(novelty))
        self.original_v_norm.append(original_norm)
        self._x7_compact()

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
        novelty_scores: list[float] = []
        if bool(getattr(self, "enable_importance", False)):
            from deltamemory.memory.bank_importance import compute_novelty

            rolling = self.state_dict()
            for row in range(n):
                score = compute_novelty(per_layer_K_batches[0][row], rolling)
                novelty_scores.append(float(score))
                rolling["M_K"] = [
                    torch.cat([k, batch[row : row + 1].detach().cpu()], dim=0)
                    for k, batch in zip(rolling["M_K"], per_layer_K_batches)
                ]
        else:
            novelty_scores = [1.0] * n
        original_norms = []
        for row in range(n):
            original_norms.append(float(sum(
                torch.linalg.vector_norm(batch[row].detach().float(), ord=2).item()
                for batch in per_layer_V_batches
            )))
        for layer in range(self.num_layers):
            d = self.head_dims[layer]
            h = self.num_kv_heads_per_layer[layer]
            for name, src in (("K", per_layer_K_batches[layer]), ("V", per_layer_V_batches[layer])):
                expected = (n, h, d)
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
        for _ in range(n):
            self._x7_global_step += 1
            self._x7_write_step.append(self._x7_global_step)
            self._x7_last_access.append(self._x7_global_step)
            self._x7_access_count.append(0)
        self.merge_counts.extend([1.0] * n)
        self.importance_scores.extend(novelty_scores)
        self.original_v_norm.extend(original_norms)
        self._x7_compact()

    def _apply_state_dict_runtime(self, sd: dict) -> None:
        self.M_K = [t.to(self.device, self.dtype) for t in sd["M_K"]]
        self.M_V = [t.to(self.device, self.dtype) for t in sd["M_V"]]
        self.fact_ids = list(sd.get("fact_ids", self.fact_ids))
        self.address_strs = list(sd.get("address_strs", self.address_strs))
        self._x7_write_step = list(sd.get("write_step", sd.get("_x7_write_step", self._x7_write_step)))
        self._x7_last_access = list(sd.get("last_access_step", sd.get("_x7_last_access", self._x7_last_access)))
        self._x7_access_count = list(sd.get("_x7_access_count", self._x7_access_count))
        self.merge_counts = [float(x) for x in sd.get("merge_counts", self.merge_counts)]
        self.importance_scores = [float(x) for x in sd.get("importance_scores", self.importance_scores)]
        self.original_v_norm = [float(x) for x in sd.get("original_v_norm", self.original_v_norm)]

    def _x7_compact(self) -> None:
        """Phase X.7: enforce bank_capacity by evicting per ``bank_evict_policy``.

        Default ``bank_capacity == 0`` ⇒ unbounded ⇒ no-op (bit-equal to v0.4).
        Per-layer ``M_K`` / ``M_V`` slices stay aligned with metadata lists.
        """
        if bool(getattr(self, "enable_decay", False)) and self.size > 0:
            from deltamemory.memory.bank_decay import apply_decay

            sd = self.state_dict()
            sd["decay_erase_threshold"] = float(getattr(self, "decay_erase_threshold", 1e-3))
            decayed = apply_decay(
                sd,
                current_step=self._x7_global_step,
                half_life=int(getattr(self, "decay_half_life", 1000) or 1000),
            )
            self._apply_state_dict_runtime(decayed)

        cap = int(getattr(self, "bank_capacity", 0) or 0)
        if cap <= 0:
            return
        n = self.size
        if n <= cap:
            return
        if bool(getattr(self, "enable_compression", False)):
            from deltamemory.memory.bank_compression import compress_bank

            threshold = int(getattr(self, "compression_threshold", 0) or cap)
            target = int(getattr(self, "compression_target_size", 0) or cap)
            if n > threshold:
                sd = self.state_dict()
                sd["compression_min_similarity"] = float(getattr(self, "compression_min_similarity", 0.90))
                compressed = compress_bank(sd, target_size=target)
                self._apply_state_dict_runtime(compressed)
                n = self.size
                if n <= cap:
                    return
        n_drop = n - cap
        policy = str(getattr(self, "bank_evict_policy", "lru"))
        if policy == "fifo":
            keep_idx = list(range(n_drop, n))
        elif policy == "lru":
            order = sorted(
                range(n),
                key=lambda i: (
                    self._x7_last_access[i],
                    self._x7_access_count[i],
                    self._x7_write_step[i],
                ),
            )
            drop = set(order[:n_drop])
            keep_idx = [i for i in range(n) if i not in drop]
        else:
            raise ValueError(
                f"unknown bank_evict_policy {policy!r}; expected 'lru' or 'fifo'"
            )
        keep_t = torch.tensor(keep_idx, dtype=torch.long)
        for layer in range(self.num_layers):
            self.M_K[layer] = self.M_K[layer].index_select(0, keep_t.to(self.M_K[layer].device))
            self.M_V[layer] = self.M_V[layer].index_select(0, keep_t.to(self.M_V[layer].device))
        self.fact_ids = [self.fact_ids[i] for i in keep_idx]
        self.address_strs = [self.address_strs[i] for i in keep_idx]
        self._x7_write_step = [self._x7_write_step[i] for i in keep_idx]
        self._x7_last_access = [self._x7_last_access[i] for i in keep_idx]
        self._x7_access_count = [self._x7_access_count[i] for i in keep_idx]
        self.merge_counts = [self.merge_counts[i] for i in keep_idx] if len(self.merge_counts) == n else []
        self.importance_scores = (
            [self.importance_scores[i] for i in keep_idx]
            if len(self.importance_scores) == n else []
        )
        self.original_v_norm = (
            [self.original_v_norm[i] for i in keep_idx]
            if len(self.original_v_norm) == n else []
        )

    def _x7_note_access(self, bank_idx: list[int]) -> None:
        """Phase X.7: forward-side hook to update access stats. No-op when capacity disabled."""
        if int(getattr(self, "bank_capacity", 0) or 0) <= 0:
            return
        self._x7_global_step += 1
        step = self._x7_global_step
        for i in bank_idx:
            if 0 <= i < len(self._x7_last_access):
                self._x7_last_access[i] = step
                self._x7_access_count[i] += 1

    def state_dict(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "head_dims": list(self.head_dims),
            "num_kv_heads_per_layer": list(self.num_kv_heads_per_layer),
            "M_K": [t.cpu() for t in self.M_K],
            "M_V": [t.cpu() for t in self.M_V],
            "fact_ids": list(self.fact_ids),
            "address_strs": list(self.address_strs),
            "bank_temperature": float(self.bank_temperature),
            "mhc_shield": bool(self.mhc_shield),
            "mhc_kappa": float(getattr(self, "mhc_kappa", 1.0)),
            "value_scale_mode": str(self.value_scale_mode),
            "value_target_rms": float(self.value_target_rms),
            "value_scale_eps": float(self.value_scale_eps),
            "bank_key_mode": str(self.bank_key_mode),
            "bank_cosine": bool(getattr(self, "bank_cosine", False)),
            "bank_topk": int(getattr(self, "bank_topk", 0) or 0),
            "bank_separate_softmax": bool(getattr(self, "bank_separate_softmax", False)),
            "bank_merge_beta": float(getattr(self, "bank_merge_beta", 1.0)),
            "merge_counts": list(self.merge_counts),
            "importance_scores": list(self.importance_scores),
            "write_step": list(self._x7_write_step),
            "last_access_step": list(self._x7_last_access),
            "original_v_norm": list(self.original_v_norm),
            "enable_compression": bool(getattr(self, "enable_compression", False)),
            "enable_decay": bool(getattr(self, "enable_decay", False)),
            "enable_importance": bool(getattr(self, "enable_importance", False)),
            "enable_tiering": bool(getattr(self, "enable_tiering", False)),
            "compression_threshold": int(getattr(self, "compression_threshold", 0) or 0),
            "compression_target_size": int(getattr(self, "compression_target_size", 0) or 0),
        }

    @classmethod
    def from_state_dict(cls, sd: dict, device="cpu", dtype=torch.bfloat16) -> "AttnNativeBank":
        bank = cls(num_layers=sd["num_layers"],
                   num_kv_heads=sd["num_kv_heads"],
                   head_dim=sd["head_dim"],
                   head_dims=list(sd.get("head_dims") or [sd["head_dim"]] * sd["num_layers"]),
                   num_kv_heads_per_layer=list(sd.get("num_kv_heads_per_layer") or [sd["num_kv_heads"]] * sd["num_layers"]),
                   device=device, dtype=dtype,
                   bank_temperature=float(sd.get("bank_temperature", 1.0)),
                   mhc_shield=bool(sd.get("mhc_shield", False)),
                   value_scale_mode=sd.get("value_scale_mode", "auto_rms_cap"),
                   value_target_rms=float(sd.get("value_target_rms", 0.5)),
                   value_scale_eps=float(sd.get("value_scale_eps", 1e-6)),
                   bank_key_mode=sd.get("bank_key_mode", "pre_rope"))
        bank.mhc_kappa = float(sd.get("mhc_kappa", 1.0))
        bank.M_K = [t.to(device, dtype) for t in sd["M_K"]]
        bank.M_V = [t.to(device, dtype) for t in sd["M_V"]]
        bank.fact_ids = list(sd["fact_ids"])
        bank.address_strs = list(sd["address_strs"])
        bank.bank_cosine = bool(sd.get("bank_cosine", False))
        bank.bank_topk = int(sd.get("bank_topk", 0) or 0)
        bank.bank_separate_softmax = bool(sd.get("bank_separate_softmax", False))
        bank.bank_merge_beta = float(sd.get("bank_merge_beta", 1.0))
        bank.merge_counts = [float(x) for x in sd.get("merge_counts", [])]
        bank.importance_scores = [float(x) for x in sd.get("importance_scores", [])]
        bank._x7_write_step = list(sd.get("write_step", []))
        bank._x7_last_access = list(sd.get("last_access_step", []))
        bank.original_v_norm = [float(x) for x in sd.get("original_v_norm", [])]
        bank.enable_compression = bool(sd.get("enable_compression", False))
        bank.enable_decay = bool(sd.get("enable_decay", False))
        bank.enable_importance = bool(sd.get("enable_importance", False))
        bank.enable_tiering = bool(sd.get("enable_tiering", False))
        bank.compression_threshold = int(sd.get("compression_threshold", 0) or 0)
        bank.compression_target_size = int(sd.get("compression_target_size", 0) or 0)
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
        record_q = getattr(ctx, "record_queries", False)

        # α=0 / empty-bank redline: if no capture, no positive bank injection,
        # and no Q recording is requested, delegate to the original module exactly.
        if not capture and not record_q and (bank is None or bank.empty or alpha <= 0.0):
            return orig_forward(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                shared_kv_states=shared_kv_states,
                **kwargs,
            )

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
                if shared_dict:
                    shared_pair = None
                    for cand_k, cand_v in shared_dict.values():
                        if cand_k.size(-1) == head_dim:
                            shared_pair = (cand_k, cand_v)
                            break
                    if shared_pair is None:
                        shared_pair = next(iter(shared_dict.values()))
                    key_states, value_states = shared_pair
                    key_states = key_states.to(q_pre.device)
                    value_states = value_states.to(q_pre.device)
                    k_dummy = torch.zeros_like(q_pre)
                    q_post_, _ = adapter.apply_rope(q_pre, k_dummy, cos, sin)
                    q_post = q_post_.transpose(1, 2)
                else:
                    # Last-resort recompute (should not happen in eager prefill,
                    # but keeps the patcher robust to future API drift).
                    k_pre = adapter.apply_k_norm(self, self.k_proj(hidden_states).view(hidden_shape))
                    v_input = self.v_proj if self.v_proj is not None else self.k_proj
                    v_post_norm = adapter.apply_v_norm(self, v_input(hidden_states).view(hidden_shape))
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
            # Gemma-4 use_alternative_attention layers have v_proj=None and
            # reuse k_proj output as values (matches transformers upstream).
            v_input = self.v_proj if self.v_proj is not None else self.k_proj
            v_post_norm = adapter.apply_v_norm(self, v_input(hidden_states).view(hidden_shape))
            q_post_, k_post_ = adapter.apply_rope(q_pre, k_pre, cos, sin)
            q_post = q_post_.transpose(1, 2)
            key_states = k_post_.transpose(1, 2)
            value_states = v_post_norm.transpose(1, 2)
            k_pre_for_capture = k_pre  # [B, T, Hkv, d]

        q_pre = q_pre.transpose(1, 2)              # [B, Hq, T, d]

        # Exp13 diagnostic: record q_pre / q_post for offline QK-only scoring.
        # Guarded by ctx.record_queries; default False keeps zero overhead.
        if record_q:
            T_now = q_pre.size(2)
            rec_pos = ctx.record_query_pos if ctx.record_query_pos is not None else (T_now - 1)
            rec_pos = max(0, min(int(rec_pos), T_now - 1))
            ctx._recorded_Q_pre[layer_idx] = q_pre[:, :, rec_pos, :].detach()    # [B, Hq, d]
            ctx._recorded_Q_post[layer_idx] = q_post[:, :, rec_pos, :].detach()  # [B, Hq, d]

        if past_key_values is not None and not is_kv_shared:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if adapter.store_full_length_kv(self):
            if shared_kv_states is not None:
                shared_kv_states[self.layer_idx] = key_states, value_states
                layer_type_key = getattr(self, "layer_type", None)
                if layer_type_key is not None:
                    shared_kv_states[layer_type_key] = key_states, value_states
            elif past_key_values is not None:
                if not hasattr(past_key_values, "shared_layers"):
                    past_key_values.shared_layers = {}
                past_key_values.shared_layers[self.layer_idx] = key_states, value_states

        # --- capture for bank write (single fact at a time) ---
        if capture and k_pre_for_capture is not None:
            T_full = k_pre_for_capture.size(1)
            pos = (T_full - 1) if capture_pos is None else int(capture_pos)
            # ``capture_bank`` carries the bank's runtime config (in particular
            # ``bank_key_mode``). Resolve it before consulting the knob below.
            capture_bank = ctx.capture_bank
            # A.1 ablation: force POST-RoPE K capture (vs the default pre-RoPE
            # for position-invariant retrieval). Hypothesis: pre-RoPE invariance
            # is necessary for bank K to score on a different read position.
            # Capture is only active in the non-shared branch where k_post_
            # is always defined alongside k_pre_for_capture.
            if getattr(ctx, "_a1_force_post_rope_capture", False):
                ctx._capture_K[layer_idx] = k_post_[:, pos, :, :].detach()
            elif getattr(capture_bank, "bank_key_mode", "pre_rope") == "post_rope":
                ctx._capture_K[layer_idx] = k_post_[:, pos, :, :].detach()
            else:
                ctx._capture_K[layer_idx] = k_pre_for_capture[:, pos, :, :].detach()  # [B, Hkv, d]
            v_captured = (value_states.transpose(1, 2)                             # [B, T, Hkv, d]
                          [:, pos, :, :].detach())
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
            bank_key_mode = getattr(bank, "bank_key_mode", "pre_rope")
            if bank_key_mode == "pre_rope":
                q_bank = q_pre
            elif bank_key_mode == "post_rope":
                q_bank = q_post
            else:
                raise ValueError(
                    f"bank_key_mode must be 'pre_rope' or 'post_rope', got {bank_key_mode!r}"
                )
            if use_cosine:
                q_cos = q_bank / (q_bank.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                k_cos = mk_e / (mk_e.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                scores_bank = torch.matmul(q_cos, k_cos.transpose(2, 3))  # [B,Hq,T,N]
            else:
                scores_bank = torch.matmul(q_bank, mk_e.transpose(2, 3)) * scaling  # [B,Hq,T,N]
            if bool(getattr(bank, "enable_importance", False)):
                from deltamemory.memory.bank_importance import importance_bias

                bias = importance_bias(bank.state_dict()).to(scores_bank.device, scores_bank.dtype)
                if bias.numel() == scores_bank.size(-1):
                    scores_bank = scores_bank * bias.reshape(1, 1, 1, -1)
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
                # Stage 17-S: mHC column cap on separate bank weights.
                if getattr(bank, "mhc_shield", False):
                    from deltamemory.memory.mhc_shield import shield_bank_weights
                    w_bank = shield_bank_weights(
                        w_bank,
                        kappa=float(getattr(bank, "mhc_kappa", 1.0)),
                    )
                beta = float(getattr(bank, "bank_merge_beta", 1.0))
                out_orig = torch.matmul(w_orig, v_repeat)
                out_bank = torch.matmul(w_bank, alpha * mv_e)
                # Diagnostics for separate-softmax branch.
                import deltamemory.diagnostics as _diag_mod  # noqa: PLC0415
                if _diag_mod._RECORDER is not None:
                    _diag_mod._RECORDER.record_bank_readout(layer_idx, out_bank, out_orig)
                attn_out = (out_orig + beta * out_bank).transpose(1, 2).contiguous()
                weights = w_orig  # for downstream sanity (unused)
            else:
                scores = torch.cat([scores_orig, scores_bank], dim=-1)
                weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q_post.dtype)
                T_orig = scores_orig.size(-1)
                # Phase X.7: forward-side LRU access tracker. Gated on
                # bank_capacity > 0 so the default path stays bit-equal.
                if int(getattr(bank, "bank_capacity", 0) or 0) > 0:
                    bank_w = weights[..., T_orig:]
                    if bank_w.size(-1) > 0:
                        idxs = bank_w.detach().argmax(dim=-1).reshape(-1).tolist()
                        bank._x7_note_access(list({int(i) for i in idxs}))
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
                        kappa=getattr(bank, "mhc_kappa", 1.0),
                    )
                out_orig = torch.matmul(weights[..., :T_orig], v_repeat)
                out_bank = torch.matmul(weights[..., T_orig:], alpha * mv_e)
                if _diag_mod._RECORDER is not None:
                    _diag_mod._RECORDER.record_bank_readout(layer_idx, out_bank, out_orig)
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
                        alpha=alpha,
                    )
                    # Write current-step norm to pending (NOT prev) -- see B1
                    with torch.no_grad():
                        lopi_state.pending_residual_norms[layer_idx] = float(
                            torch.linalg.vector_norm(out_orig, ord=2, dim=-1).mean().item()
                        )
                # Stage 17 (v3.4): optional merged-path beta gate.
                # When bank_merge_beta != 1.0 the bank read-out is scaled
                # before addition, mirroring legacy residual smoothing:
                #   x' = x + beta * out_bank
                # Default 1.0 keeps v3.2 bit-equal.
                merged_beta = float(getattr(bank, "bank_merge_beta", 1.0))
                if merged_beta != 1.0:
                    out_bank = out_bank * merged_beta
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

    def __init__(
        self,
        model,
        adapter: "ArchAdapter | None" = None,
        *,
        enable_compression: bool = False,
        enable_decay: bool = False,
        enable_importance: bool = False,
        enable_tiering: bool = False,
        compression_threshold: int = 0,
    ):
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
        # Exp13 addition: per-layer Q recording for QK-only diagnostics.
        # Default False → strictly additive, no behavior change. When True,
        # the patched forward records q_pre (and q_post) at every layer for
        # the configured token position(s) into ``_recorded_Q_pre/_recorded_Q_post``.
        self.record_queries: bool = False
        self.record_query_pos: int | None = None  # None → last real token (per row)
        self._recorded_Q_pre: list[torch.Tensor | None] = [None] * self.num_layers
        self._recorded_Q_post: list[torch.Tensor | None] = [None] * self.num_layers
        self.enable_compression = bool(enable_compression)
        self.enable_decay = bool(enable_decay)
        self.enable_importance = bool(enable_importance)
        self.enable_tiering = bool(enable_tiering)
        self.compression_threshold = int(compression_threshold or 0)

    def _apply_smart_flags(self, bank: AttnNativeBank) -> None:
        for name in ("enable_compression", "enable_decay", "enable_importance", "enable_tiering"):
            if bool(getattr(self, name, False)):
                setattr(bank, name, True)
        if self.compression_threshold > 0:
            bank.compression_threshold = self.compression_threshold

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
        self._apply_smart_flags(bank)
        self.bank, self.alpha = bank, float(alpha)
        try:
            yield
        finally:
            self.bank, self.alpha = prev_bank, prev_alpha

    @contextmanager
    def recording_queries(self, capture_pos: int | None = None):
        """Exp13 QK-only diagnostic: record per-layer Q at a single token pos.

        Pure recorder — does not inject or modify outputs.  Sets
        ``record_queries=True`` and pins ``record_query_pos`` for the duration.
        On exit, recorded tensors remain in ``_recorded_Q_pre/_recorded_Q_post``
        for the caller to consume.
        """
        prev_flag = self.record_queries
        prev_pos = self.record_query_pos
        self.record_queries = True
        self.record_query_pos = capture_pos
        self._recorded_Q_pre = [None] * self.num_layers
        self._recorded_Q_post = [None] * self.num_layers
        try:
            yield
        finally:
            self.record_queries = prev_flag
            self.record_query_pos = prev_pos


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
    num_kv_heads_per_layer: list[int] = []
    default_kv = cfg.num_key_value_heads
    for sa in patcher_probe.attn_modules:
        d = getattr(sa, "head_dim", None) or cfg_head_dim
        head_dims.append(int(d))
        # Probe per-layer kv head count: prefer attribute, else derive from k_proj.out_features / head_dim.
        h = getattr(sa, "num_key_value_heads", None) or getattr(sa, "num_kv_heads", None)
        if h is None and getattr(sa, "k_proj", None) is not None and d:
            try:
                h = int(sa.k_proj.out_features // int(d))
            except Exception:
                h = None
        num_kv_heads_per_layer.append(int(h) if h is not None else int(default_kv))

    return AttnNativeBank(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=int(cfg_head_dim),
        head_dims=head_dims,
        num_kv_heads_per_layer=num_kv_heads_per_layer,
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
            if kc is None:
                target = (bank.num_kv_heads_per_layer[layer], bank.head_dims[layer])
                for cand_k, cand_v in zip(patcher._capture_K, patcher._capture_V):
                    if cand_k is not None and tuple(cand_k.shape[1:]) == target:
                        kc, vc = cand_k, cand_v
                        break
            if kc is None or vc is None:
                raise RuntimeError(f"write_fact: no captured KV source for shared layer {layer}")
            K_per_layer.append(kc[0])
            V_per_layer.append(vc[0])
        site_fact_id = (
            fact_id if len(sites) == 1 else f"{fact_id}@{site.role}"
        )
        patcher._apply_smart_flags(bank)
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
