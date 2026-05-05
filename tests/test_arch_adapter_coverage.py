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
import torch.nn as nn

from deltamemory.memory.arch_adapter import (
    Gemma2Adapter,
    Gemma3Adapter,
    Gemma4Adapter,
    Glm4Adapter,
    LlamaAdapter,
    Phi3Adapter,
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
    ("Phi3Attention", Phi3Adapter),
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
