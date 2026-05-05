"""A4 — cross-model adapter coverage audit.

Confirms that every modern HF attention class we plan to support routes to a
specific :class:`ArchAdapter` via :func:`pick_adapter`, instead of hitting the
``NotImplementedError`` fallback. Adding a new family means: add a subclass,
add it to ``_REGISTRY``, and extend ``EXPECTED`` below.

These tests use *minimal* nn.Module stubs so they do not require gated model
weights or HF downloads — they only assert that class-name dispatch works.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deltamemory.memory.arch_adapter import (
    Gemma2Adapter,
    Gemma3Adapter,
    Gemma4Adapter,
    Glm4Adapter,
    LlamaAdapter,
    Qwen3Adapter,
    pick_adapter,
)


def _stub(class_name: str) -> nn.Module:
    """Build an empty nn.Module whose ``type(...).__name__`` equals
    ``class_name`` — the only signal that ``ArchAdapter.matches`` reads."""
    cls = type(class_name, (nn.Module,), {})
    return cls()


EXPECTED = [
    # (HF attention class name, expected adapter type)
    ("Gemma3nTextAttention", Gemma4Adapter),
    ("Gemma4Attention", Gemma4Adapter),
    ("Gemma3Attention", Gemma3Adapter),
    ("Gemma2Attention", Gemma2Adapter),
    ("Qwen3Attention", Qwen3Adapter),
    ("LlamaAttention", LlamaAdapter),
    ("Qwen2Attention", LlamaAdapter),
    ("MistralAttention", LlamaAdapter),
    ("Glm4Attention", Glm4Adapter),
]


@pytest.mark.parametrize("class_name,adapter_type", EXPECTED)
def test_pick_adapter_routes_modern_families(class_name, adapter_type):
    attn = _stub(class_name)
    picked = pick_adapter(attn)
    assert isinstance(picked, adapter_type), (
        f"{class_name} routed to {type(picked).__name__}, "
        f"expected {adapter_type.__name__}"
    )


def test_unmapped_class_still_raises():
    attn = _stub("MysteryAttention")
    with pytest.raises(NotImplementedError):
        pick_adapter(attn)


def test_gemma3_does_not_collide_with_gemma3n():
    """Regression: ensure plain Gemma3Attention does NOT pick Gemma4Adapter
    (which historically matched 'Gemma3n' substring) — this was the W.4 PREREG
    blocker for gemma-3-270m / gemma-3-1b-it on the bank patcher path."""
    g3 = _stub("Gemma3Attention")
    assert isinstance(pick_adapter(g3), Gemma3Adapter)
    g3n = _stub("Gemma3nTextAttention")
    assert isinstance(pick_adapter(g3n), Gemma4Adapter)


def test_gemma3_norms_required():
    """Regression: Gemma3Adapter must enforce q_norm and k_norm presence.

    Gemma-3 natively applies q_norm and k_norm after proj+reshape, before rope.
    Our patched forward (attn_native_bank.py lines ~403-417) respects this
    order. If either norm is missing, applying it is a logic error: patched
    forward will call adapter.apply_q_norm / apply_k_norm and expect a callable.
    """
    # Stub without q_norm
    class Gemma3AttentionNoQNorm(nn.Module):
        pass
    attn_no_q = Gemma3AttentionNoQNorm()
    attn_no_q.__class__.__name__ = "Gemma3Attention"
    adapter = Gemma3Adapter()
    dummy_q = torch.zeros(1, 8, 512)
    with pytest.raises(AssertionError, match="q_norm"):
        adapter.apply_q_norm(attn_no_q, dummy_q)

    # Stub without k_norm
    class Gemma3AttentionNoKNorm(nn.Module):
        pass
    attn_no_k = Gemma3AttentionNoKNorm()
    attn_no_k.__class__.__name__ = "Gemma3Attention"
    dummy_k = torch.zeros(1, 8, 512)
    with pytest.raises(AssertionError, match="k_norm"):
        adapter.apply_k_norm(attn_no_k, dummy_k)


def test_phi3_remains_unmapped():
    """Regression: Phi3Adapter removed because HF Phi3Attention uses fused
    qkv_proj, not separate q/k/v_proj. Patched forward (attn_native_bank.py
    lines ~403-417) assumes separate projections (self.q_proj, self.k_proj,
    self.v_proj), so matching Phi3Attention would crash at runtime.

    This test confirms Phi3Attention is not mapped to any adapter and raises
    NotImplementedError, blocking silent runtime failure. A future PR can add
    a fused-qkv codepath if Phi3 is adopted.
    """
    phi3 = _stub("Phi3Attention")
    with pytest.raises(NotImplementedError, match="No ArchAdapter matches"):
        pick_adapter(phi3)
