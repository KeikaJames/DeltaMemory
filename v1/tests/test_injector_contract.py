"""Architectural contract tests for the injector layer-locator and α=0 redline.

Single source of truth: ``deltamemory.memory._layer_locator.get_decoder_layers``.
This test file enforces:
  1. CAA / SCAR delegate to the shared locator (importable, identical behaviour).
  2. Both injectors honour the α=0 bit-equal redline through identical
     ``get_decoder_layers`` paths and identical model shape.
  3. The locator handles both the modern ``model.layers`` path (Qwen3 / Llama /
     Gemma / GLM family) and the legacy GPT-2 ``transformer.h`` fallback.
  4. The locator raises a clear RuntimeError on a model without decoder layers.

If a future injector is added, its α=0 bit-equal contract should be added to
``INJECTOR_FACTORIES`` below; the parametrised redline test will cover it.

Author: KeikaJames, 2026-05-06.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from deltamemory.memory._layer_locator import get_decoder_layers
from deltamemory.memory.caa_injector import CAAConfig, CAAInjector
from deltamemory.memory.scar_injector import SCARInjector


# ---------------------------------------------------------------------------
# Tiny model fixtures (no network) — Qwen3 is the modern flagship arch
# (model.layers / self_attn.o_proj / GQA / RoPE), GPT-2 covers the legacy
# transformer.h fallback path.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_qwen3():
    """Tiny Qwen3 — modern flagship architecture (random weights, no download)."""
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def tiny_gpt2_legacy():
    """Tiny GPT-2 — exercises the legacy ``transformer.h`` locator path only."""
    cfg = GPT2Config(
        vocab_size=256,
        n_embd=64,
        n_layer=4,
        n_head=4,
        n_positions=128,
        n_inner=128,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).eval()


# ---------------------------------------------------------------------------
# Locator tests
# ---------------------------------------------------------------------------


def test_locator_finds_qwen3_model_layers(tiny_qwen3):
    """get_decoder_layers locates ``model.layers`` on the modern flagship arch."""
    layers = get_decoder_layers(tiny_qwen3)
    assert len(layers) == 4
    # Modern arch exposes ``self_attn.o_proj`` (used by SCAR's hook target).
    assert hasattr(layers[0], "self_attn")
    assert hasattr(layers[0].self_attn, "o_proj")


def test_locator_finds_gpt2_transformer_h_legacy(tiny_gpt2_legacy):
    """get_decoder_layers locates the legacy ``transformer.h`` path."""
    layers = get_decoder_layers(tiny_gpt2_legacy)
    assert len(layers) == 4
    assert hasattr(layers[0], "attn")
    assert hasattr(layers[0], "mlp")


def test_locator_caa_and_scar_agree(tiny_qwen3):
    """CAA._get_decoder_layers and SCAR._get_decoder_layers produce identical lists."""
    cfg = CAAConfig(inject_layer=0, alpha=0.0)
    caa = CAAInjector(tiny_qwen3, cfg)
    scar = SCARInjector(tiny_qwen3, alpha=0.0, layers=[0])
    caa_layers = caa._get_decoder_layers()
    scar_layers = scar._get_decoder_layers()
    canonical = get_decoder_layers(tiny_qwen3)
    assert len(caa_layers) == len(scar_layers) == len(canonical) == 4
    for a, b, c in zip(caa_layers, scar_layers, canonical):
        assert a is b is c


def test_locator_raises_on_bare_module():
    """A plain nn.Module with no decoder layers must raise RuntimeError."""
    bare = nn.Linear(8, 8)
    with pytest.raises(RuntimeError, match="could not locate decoder layers"):
        get_decoder_layers(bare)


# ---------------------------------------------------------------------------
# α=0 bit-equal redline — parametrised across all registered injectors,
# exercised on the modern flagship arch.
# ---------------------------------------------------------------------------


def _make_caa(model):
    cfg = CAAConfig(inject_layer=0, alpha=0.0)
    inj = CAAInjector(model, cfg)
    inj.steering_vector = torch.randn(model.config.hidden_size)
    return inj


def _make_scar(model):
    return SCARInjector(model, alpha=0.0, layers=[0], k=2)


INJECTOR_FACTORIES = [
    pytest.param(_make_caa, id="caa"),
    pytest.param(_make_scar, id="scar"),
]


@pytest.mark.parametrize("factory", INJECTOR_FACTORIES)
def test_alpha_zero_is_bit_equal_contract(tiny_qwen3, factory):
    """Every injector with α=0 must produce logits identical to no-injection."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        baseline = tiny_qwen3(input_ids).logits.clone()

    inj = factory(tiny_qwen3)
    with inj:
        with torch.no_grad():
            injected = tiny_qwen3(input_ids).logits

    max_abs_diff = (baseline - injected).abs().max().item()
    assert max_abs_diff == 0.0, (
        f"α=0 bit-equal redline broken: max-abs-diff={max_abs_diff}"
    )
