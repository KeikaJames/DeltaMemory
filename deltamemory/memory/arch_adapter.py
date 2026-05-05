"""Architecture adapters for the attn-native bank patcher.

Mneme's red line: the base LLM weights are frozen. Adapters only **read**
per-family attention conventions (q/k/v norm presence, KV-sharing, RoPE
function) and never modify weights.

Each adapter is a stateless object that the patched forward consults at every
attention call. Adding a new family means writing a new ``ArchAdapter`` subclass
plus a class-name match rule; the patched forward itself stays generic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

@dataclass
class ArchAdapter:
    """Minimal interface every supported family must satisfy.

    Subclasses override the methods that differ from the no-op defaults below.
    The patched forward only ever calls these methods; it never imports a
    family-specific module directly.
    """

    name: str = "base"

    # --- default α calibration (per-arch) -----------------------------------
    # Empirically calibrated injection scale for the bank K/V branch on this
    # family.  Different architectures have very different V-activation ranges
    # (e.g. Gemma3n applies v_norm so V is small → α≈1 is fine; Qwen3 has no
    # v_norm so V is larger → α=1 destroys the logits, α≈0.05 preserves
    # them).  Conservation tests at α=0/empty bank do NOT catch this because
    # they never exercise the injection path.  See
    # ``transcripts/v31_intervention/CROSS_ARCH_REPORT.md`` for the data.
    default_alpha: float = 1.0

    # --- class-name match (used by AttnNativePatcher to auto-pick) -----------
    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        """Return True if this adapter handles ``attn_module``'s class."""
        return False

    # --- norms (default: identity) -------------------------------------------
    def apply_q_norm(self, attn: nn.Module, q: torch.Tensor) -> torch.Tensor:
        fn = getattr(attn, "q_norm", None)
        return fn(q) if callable(fn) else q

    def apply_k_norm(self, attn: nn.Module, k: torch.Tensor) -> torch.Tensor:
        fn = getattr(attn, "k_norm", None)
        return fn(k) if callable(fn) else k

    def apply_v_norm(self, attn: nn.Module, v: torch.Tensor) -> torch.Tensor:
        fn = getattr(attn, "v_norm", None)
        return fn(v) if callable(fn) else v

    def has_native_v_norm(self, attn: nn.Module) -> bool:
        """Whether this attention module normalizes V in the native forward."""
        return callable(getattr(attn, "v_norm", None))

    # --- RoPE (must override) ------------------------------------------------
    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(f"{type(self).__name__}.apply_rope")

    # --- KV-sharing (default: never shared) ----------------------------------
    def is_kv_shared(self, attn: nn.Module) -> bool:
        return bool(getattr(attn, "is_kv_shared_layer", False))

    def kv_shared_index(self, attn: nn.Module) -> int | None:
        return getattr(attn, "kv_shared_layer_index", None)

    def store_full_length_kv(self, attn: nn.Module) -> bool:
        return bool(getattr(attn, "store_full_length_kv", False))

    # --- repeat_kv (default: standard GQA repeat from transformers) ----------
    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return x
        b, h, t, d = x.shape
        return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


# ---------------------------------------------------------------------------
# Gemma-4 (production)
# ---------------------------------------------------------------------------

class Gemma4Adapter(ArchAdapter):
    """Adapter for Gemma3nTextAttention (used by gemma-4-E2B / -31B-it).

    Has q/k/v_norm; uses KV-sharing on a subset of layers; RoPE pulled from
    ``transformers.models.gemma4.modeling_gemma4.apply_rotary_pos_emb``.
    """

    def __init__(self):
        super().__init__(name="gemma4", default_alpha=1.0)

    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        return "Gemma4" in type(attn_module).__name__ or "Gemma3n" in type(attn_module).__name__

    def apply_rope(self, q, k, cos, sin):
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
        # Gemma-4 uses unsqueeze_dim=2 (per-head broadcast).
        q2 = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)
        k2 = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2)
        return q2, k2

    # repeat_kv: prefer the canonical implementation when available so we are
    # bit-equal to upstream Gemma-4.
    def repeat_kv(self, x, n_rep):
        from transformers.models.gemma4.modeling_gemma4 import repeat_kv as _rk
        return _rk(x, n_rep)


# ---------------------------------------------------------------------------
# Qwen-3
# ---------------------------------------------------------------------------

class Qwen3Adapter(ArchAdapter):
    """Adapter for Qwen3Attention (used by Qwen/Qwen3-4B-Instruct etc.).

    * q_norm / k_norm present, no v_norm.
    * No KV-sharing.
    * RoPE base = 1e6; uses transformers Qwen3 apply_rotary_pos_emb.
    """

    def __init__(self):
        super().__init__(name="qwen3", default_alpha=0.05)

    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        return "Qwen3" in type(attn_module).__name__

    def is_kv_shared(self, attn):
        return False

    def store_full_length_kv(self, attn):
        return False

    def apply_rope(self, q, k, cos, sin):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        # Qwen3 upstream expects q/k in (B, H, T, D) with unsqueeze_dim=1.
        # Our patched forward holds them in (B, T, H, D); transpose, apply,
        # transpose back to keep the contract uniform.
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        q2, k2 = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=1)
        return q2.transpose(1, 2), k2.transpose(1, 2)


# ---------------------------------------------------------------------------
# Llama family (Llama-2/3, DeepSeek-R1-Distill-Qwen-32B is Qwen2-architecture
# but distilled — we route Qwen2 and Llama through this adapter)
# ---------------------------------------------------------------------------

class LlamaAdapter(ArchAdapter):
    """Adapter for LlamaAttention / Qwen2Attention / MistralAttention.

    No q/k/v norms (defaults inherited as identity), no KV-sharing, standard
    RoPE.
    """

    def __init__(self):
        # Qwen2 family (incl. DeepSeek-R1-Distill-Qwen-32B) calibrated at 0.05
        # on GB10 cuda bf16; raw Llama-2/3 with v_norm-less V also defaults
        # here.  Override per-model if needed.
        super().__init__(name="llama", default_alpha=0.05)

    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        n = type(attn_module).__name__
        return any(tag in n for tag in ("LlamaAttention", "Qwen2Attention", "MistralAttention"))

    def apply_rope(self, q, k, cos, sin):
        # Llama-family helper expects (B, H, T, D); our forward holds (B, T, H, D).
        try:
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        except Exception:  # pragma: no cover
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        q2, k2 = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=1)
        return q2.transpose(1, 2), k2.transpose(1, 2)


# ---------------------------------------------------------------------------
# GLM-4
# ---------------------------------------------------------------------------

class Glm4Adapter(ArchAdapter):
    """Adapter for Glm4Attention (THUDM/glm-4-9b-chat).

    GLM-4 uses partial RoPE (only the first half of head_dim is rotated) and
    no q/k/v norms in the attention module itself.
    """

    def __init__(self):
        super().__init__(name="glm4", default_alpha=0.05)

    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        n = type(attn_module).__name__
        return "Glm4" in n  # HF-native GLM-4 only; ChatGLM (trust_remote_code) uses different RoPE

    def apply_rope(self, q, k, cos, sin):
        from transformers.models.glm4.modeling_glm4 import apply_rotary_pos_emb
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        q2, k2 = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=1)
        return q2.transpose(1, 2), k2.transpose(1, 2)


# ---------------------------------------------------------------------------
# Gemma-3 (gemma-3-270m / gemma-3-1b-it; used in W.4 PREREG)
# ---------------------------------------------------------------------------

class Gemma3Adapter(ArchAdapter):
    """Adapter for Gemma3Attention (NOT Gemma3n / NOT Gemma-4).

    Plain Gemma-3 (gemma-3-270m, gemma-3-1b-it) requires explicit q_norm and
    k_norm application. Unlike the base adapter which silently skips missing
    norms, this adapter enforces their presence and applies them in the native
    order: proj → reshape → q/k_norm → rope. (Gemma-3 applies q_norm/k_norm
    AFTER reshape, BEFORE rope; the patched forward at attn_native_bank.py
    lines ~403-417 respects this order.)

    No KV-sharing, standard RoPE with ``unsqueeze_dim=1``. Distinct module
    path from Gemma-4 (``modeling_gemma4``); this adapter imports from
    ``modeling_gemma3``.
    """

    def __init__(self):
        super().__init__(name="gemma3", default_alpha=0.05)

    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        n = type(attn_module).__name__
        return n == "Gemma3Attention"

    def apply_q_norm(self, attn: nn.Module, q: torch.Tensor) -> torch.Tensor:
        fn = getattr(attn, "q_norm", None)
        assert callable(fn), (
            f"Gemma3Adapter requires {type(attn).__name__}.q_norm to be callable "
            f"(native Gemma-3 applies q_norm after reshape, before rope)"
        )
        return fn(q)

    def apply_k_norm(self, attn: nn.Module, k: torch.Tensor) -> torch.Tensor:
        fn = getattr(attn, "k_norm", None)
        assert callable(fn), (
            f"Gemma3Adapter requires {type(attn).__name__}.k_norm to be callable "
            f"(native Gemma-3 applies k_norm after reshape, before rope)"
        )
        return fn(k)

    def apply_rope(self, q, k, cos, sin):
        from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        q2, k2 = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=1)
        return q2.transpose(1, 2), k2.transpose(1, 2)


# ---------------------------------------------------------------------------
# Gemma-2 (legacy completeness)
# ---------------------------------------------------------------------------

class Gemma2Adapter(ArchAdapter):
    """Adapter for Gemma2Attention. Same shape as Gemma-3 / Llama (no norms,
    standard RoPE, ``unsqueeze_dim=1``)."""

    def __init__(self):
        super().__init__(name="gemma2", default_alpha=0.05)

    @classmethod
    def matches(cls, attn_module: nn.Module) -> bool:
        return type(attn_module).__name__ == "Gemma2Attention"

    def apply_rope(self, q, k, cos, sin):
        from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        q2, k2 = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=1)
        return q2.transpose(1, 2), k2.transpose(1, 2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: list[type[ArchAdapter]] = [
    Gemma4Adapter,
    Gemma3Adapter,
    Gemma2Adapter,
    Qwen3Adapter,
    LlamaAdapter,
    Glm4Adapter,
]


def pick_adapter(attn_module: nn.Module) -> ArchAdapter:
    """Return the first adapter in the registry whose ``matches`` returns True.

    Raises :class:`NotImplementedError` with the offending class name if no
    adapter matches; callers can catch this and either install a new adapter
    or fall back.
    """
    for cls in _REGISTRY:
        if cls.matches(attn_module):
            return cls()
    raise NotImplementedError(
        f"No ArchAdapter matches attention class {type(attn_module).__name__!r}. "
        f"Register a new ArchAdapter subclass in deltamemory.memory.arch_adapter."
    )


def register_adapter(cls: type[ArchAdapter]) -> type[ArchAdapter]:
    """Decorator/function to add a new adapter to the registry."""
    if cls not in _REGISTRY:
        _REGISTRY.insert(0, cls)  # newer registrations win
    return cls


__all__ = [
    "ArchAdapter",
    "Gemma4Adapter",
    "Gemma3Adapter",
    "Gemma2Adapter",
    "Qwen3Adapter",
    "LlamaAdapter",
    "Glm4Adapter",
    "pick_adapter",
    "register_adapter",
]
