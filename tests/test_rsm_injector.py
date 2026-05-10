"""Tests for Residual Stream Memory."""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from deltamemory.memory.rsm_injector import RSMConfig, RSMInjector, RSMMemoryBank


_VOCAB = 128
_LAYERS = 3
_HIDDEN = 32
_HEADS = 4


@pytest.fixture()
def tiny_gpt2():
    cfg = GPT2Config(
        vocab_size=_VOCAB,
        n_embd=_HIDDEN,
        n_layer=_LAYERS,
        n_head=_HEADS,
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


def test_capture_shape(tiny_gpt2):
    rsm = RSMInjector(tiny_gpt2)
    memory = rsm.capture(_ids())
    assert memory.shape == (_LAYERS, _HIDDEN)
    assert memory.dtype == torch.float32


def test_eta_zero_is_bit_equal_with_nonempty_bank(tiny_gpt2):
    ids = _ids(1)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.0, theta=-1.0))
    memory = rsm.capture(ids)
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])

    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, _diag = rsm.forward_with_memory(bank, input_ids=ids)
        injected = out.logits.detach().clone()

    assert torch.equal(base, injected)


def test_empty_bank_is_bit_equal(tiny_gpt2):
    ids = _ids(2)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.5, theta=-1.0))
    empty = RSMMemoryBank(torch.empty(0, _LAYERS, _HIDDEN), [])

    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, diag = rsm.forward_with_memory(empty, input_ids=ids)
        injected = out.logits.detach().clone()

    assert torch.equal(base, injected)
    assert diag["rsm_activation_rate"] == 0.0


def test_gate_threshold_suppresses_injection(tiny_gpt2):
    ids = _ids(3)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=1.0, theta=2.0))
    memory = rsm.capture(ids)
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])

    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, diag = rsm.forward_with_memory(bank, input_ids=ids)
        injected = out.logits.detach().clone()

    assert torch.equal(base, injected)
    assert diag["rsm_activation_rate"] == 0.0


def test_gate_off_changes_output(tiny_gpt2):
    ids = _ids(4)
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.25, theta=2.0, gate_off=True))
    memory = rsm.capture(ids)
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])

    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
        out, diag = rsm.forward_with_memory(bank, input_ids=ids)
        injected = out.logits.detach().clone()

    assert not torch.equal(base, injected)
    assert diag["rsm_activation_rate"] == 1.0


def test_shuffled_layers_preserves_shape_and_norms(tiny_gpt2):
    rsm = RSMInjector(tiny_gpt2)
    memory = rsm.capture(_ids(5))
    bank = RSMMemoryBank(memory.unsqueeze(0), ["fact_0"])
    shuffled = bank.shuffled_layers(seed=123)

    assert shuffled.memories.shape == bank.memories.shape
    assert torch.allclose(
        shuffled.memories.norm(dim=-1).sort().values,
        bank.memories.norm(dim=-1).sort().values,
    )


def test_hooks_detach_after_forward(tiny_gpt2):
    rsm = RSMInjector(tiny_gpt2, RSMConfig(eta=0.2, theta=-1.0))
    ids = _ids(6)
    bank = RSMMemoryBank(rsm.capture(ids).unsqueeze(0), ["fact_0"])
    before = [len(layer._forward_hooks) for layer in rsm.layers]
    rsm.forward_with_memory(bank, input_ids=ids)
    after = [len(layer._forward_hooks) for layer in rsm.layers]
    assert after == before
