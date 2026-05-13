"""Smoke test for MLPGatedInjector — identity-safety + shape sanity."""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from deltamemory.memory.mlp_gated_injector import (
    MLPGatedConfig,
    MLPGatedInjector,
    MLPGatedRouter,
    MLPMemoryBank,
)


_VOCAB = 128
_LAYERS = 3
_HIDDEN = 32


@pytest.fixture()
def tiny_gpt2():
    cfg = GPT2Config(
        vocab_size=_VOCAB,
        n_embd=_HIDDEN,
        n_layer=_LAYERS,
        n_head=4,
        n_positions=64,
        n_inner=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).eval()


def _ids(seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB, (1, 12), generator=gen)


def test_capture_returns_correct_shapes(tiny_gpt2):
    inj = MLPGatedInjector(tiny_gpt2)
    ids = _ids()
    K, V = inj.capture_at_pos(ids, pos=ids.shape[1] - 1)
    assert K.shape == (_LAYERS, _HIDDEN)
    assert V.shape == (_LAYERS, _HIDDEN)


def test_empty_bank_is_identity(tiny_gpt2):
    inj = MLPGatedInjector(tiny_gpt2)
    ids = _ids()
    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids).logits

    router = MLPGatedRouter(num_layers=_LAYERS, d_model=_HIDDEN)
    bank = MLPMemoryBank(
        M_K=[torch.empty(0, _HIDDEN) for _ in range(_LAYERS)],
        M_V=[torch.empty(0, _HIDDEN) for _ in range(_LAYERS)],
        fact_ids=[],
    )
    inj.install(bank, router, last_token_idx=ids.shape[1] - 1)
    try:
        with torch.no_grad():
            out = tiny_gpt2(input_ids=ids).logits
    finally:
        inj.remove()
    torch.testing.assert_close(base, out)


def test_gate_off_is_identity_with_nonempty_bank(tiny_gpt2):
    inj = MLPGatedInjector(tiny_gpt2, MLPGatedConfig(gate_mode="off"))
    ids = _ids(1)
    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids).logits

    router = MLPGatedRouter(num_layers=_LAYERS, d_model=_HIDDEN)
    bank = MLPMemoryBank(
        M_K=[torch.randn(4, _HIDDEN) for _ in range(_LAYERS)],
        M_V=[torch.randn(4, _HIDDEN) for _ in range(_LAYERS)],
        fact_ids=["a", "b", "c", "d"],
    )
    inj.install(bank, router, last_token_idx=ids.shape[1] - 1)
    try:
        with torch.no_grad():
            out = tiny_gpt2(input_ids=ids).logits
    finally:
        inj.remove()
    torch.testing.assert_close(base, out)


def test_init_gate_bias_near_identity(tiny_gpt2):
    """Default gate bias = -4 ⇒ sigmoid(-4) ≈ 0.018 ⇒ small init perturbation."""
    inj = MLPGatedInjector(tiny_gpt2, MLPGatedConfig(gate_mode="learned"))
    ids = _ids(2)
    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids).logits

    router = MLPGatedRouter(num_layers=_LAYERS, d_model=_HIDDEN)
    bank = MLPMemoryBank(
        M_K=[torch.randn(4, _HIDDEN) for _ in range(_LAYERS)],
        M_V=[torch.randn(4, _HIDDEN) for _ in range(_LAYERS)],
        fact_ids=["a", "b", "c", "d"],
    )
    inj.install(bank, router, last_token_idx=ids.shape[1] - 1)
    try:
        with torch.no_grad():
            out = tiny_gpt2(input_ids=ids).logits
    finally:
        inj.remove()
    rel = (out - base).abs().max() / base.abs().max().clamp_min(1e-6)
    assert rel < 1.0, f"Init-time perturbation too large: {rel.item():.4f}"

