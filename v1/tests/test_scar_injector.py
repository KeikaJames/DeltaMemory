"""Tests for SCARInjector."""
from __future__ import annotations

import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from deltamemory.memory.scar_injector import SCARInjector


_VOCAB = 128
_LAYERS = 3
_HIDDEN = 32
_HEADS = 4


class TinyTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt", truncation: bool = True, max_length: int = 64):
        ids = [min(ord(ch), _VOCAB - 1) for ch in text]
        if truncation:
            ids = ids[:max_length]
        if not ids:
            ids = [0]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@pytest.fixture()
def tiny_gpt2():
    torch.manual_seed(1234)
    cfg = GPT2Config(
        vocab_size=_VOCAB,
        n_embd=_HIDDEN,
        n_layer=_LAYERS,
        n_head=_HEADS,
        n_positions=320,
        n_inner=64,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).eval(), TinyTokenizer()


POS_PROMPTS = [
    "truthful positive memory alpha",
    "honest helpful answer beta",
    "correct stable recall gamma",
    "faithful factual response delta",
    "precise grounded statement epsilon",
    "reliable answer with context zeta",
    "consistent memory trace eta",
    "validated response theta",
]
NEG_PROMPTS = [
    "false negative drift alpha",
    "dishonest harmful answer beta",
    "incorrect unstable recall gamma",
    "unfaithful fictional response delta",
    "vague ungrounded statement epsilon",
    "unreliable answer without context zeta",
    "inconsistent memory trace eta",
    "invalid response theta",
]


def _random_ids(tokens: int = 256) -> torch.Tensor:
    return torch.randint(0, _VOCAB, (1, tokens), generator=torch.Generator().manual_seed(42))


def test_scar_alpha_zero_bit_equal(tiny_gpt2):
    model, tokenizer = tiny_gpt2
    injector = SCARInjector(model, alpha=0.0, layers=[1], k=2)
    injector.calibrate(POS_PROMPTS[:4], NEG_PROMPTS[:4], tokenizer, max_n=4)
    input_ids = _random_ids(256)

    with torch.no_grad():
        baseline = model(input_ids=input_ids, use_cache=False).logits.detach().clone()
    injector.attach()
    try:
        with torch.no_grad():
            injected = model(input_ids=input_ids, use_cache=False).logits.detach().clone()
    finally:
        injector.detach()

    max_diff = (baseline - injected).abs().max().item()
    assert max_diff == 0.0


def test_scar_basis_shape(tiny_gpt2):
    model, tokenizer = tiny_gpt2
    injector = SCARInjector(model, alpha=0.5, layers=[0], k=2)
    injector.calibrate(POS_PROMPTS, NEG_PROMPTS, tokenizer, max_n=8)

    basis = injector.basis[0]
    assert basis.shape == (_HIDDEN, 2)
    assert torch.allclose(basis.T @ basis, torch.eye(2), atol=1e-4)


def test_scar_projection_idempotent(tiny_gpt2):
    model, tokenizer = tiny_gpt2
    injector = SCARInjector(model, alpha=0.5, layers=[0], k=2)
    injector.calibrate(POS_PROMPTS, NEG_PROMPTS, tokenizer, max_n=8)

    basis = injector.basis[0]
    projector = basis @ basis.T
    x = torch.randn(_HIDDEN, generator=torch.Generator().manual_seed(7))
    assert torch.allclose(projector @ projector @ x, projector @ x, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_scar_cuda_equiv(tiny_gpt2):
    cpu_model, tokenizer = tiny_gpt2
    cuda_model = GPT2LMHeadModel(cpu_model.config).eval().cuda()
    cuda_model.load_state_dict(copy.deepcopy(cpu_model.state_dict()))

    cpu_injector = SCARInjector(cpu_model, alpha=0.5, layers=[1], k=2)
    cuda_injector = SCARInjector(cuda_model, alpha=0.5, layers=[1], k=2)
    cpu_injector.calibrate(POS_PROMPTS[:4], NEG_PROMPTS[:4], tokenizer, max_n=4)
    cuda_injector.calibrate(POS_PROMPTS[:4], NEG_PROMPTS[:4], tokenizer, max_n=4)

    input_ids = _random_ids(64)
    with cpu_injector, torch.no_grad():
        cpu_logits = cpu_model(input_ids=input_ids, use_cache=False).logits.detach().cpu()
    with cuda_injector, torch.no_grad():
        cuda_logits = cuda_model(input_ids=input_ids.cuda(), use_cache=False).logits.detach().cpu()

    assert torch.allclose(cpu_logits, cuda_logits, rtol=1e-2, atol=1e-2)


def test_scar_empty_bank_noop(tiny_gpt2):
    model, _ = tiny_gpt2
    injector = SCARInjector(model, alpha=0.5, layers=[1], k=2)
    input_ids = _random_ids(256)

    with torch.no_grad():
        baseline = model(input_ids=input_ids, use_cache=False).logits.detach().clone()
    injector.attach()
    try:
        with torch.no_grad():
            injected = model(input_ids=input_ids, use_cache=False).logits.detach().clone()
    finally:
        injector.detach()

    assert torch.equal(baseline, injected)


def test_scar_layer_hook_path(tiny_gpt2):
    model, tokenizer = tiny_gpt2
    layers = [0, 2]
    injector = SCARInjector(model, alpha=0.5, layers=layers, k=2)
    injector.calibrate(POS_PROMPTS[:4], NEG_PROMPTS[:4], tokenizer, max_n=4)
    modules = [model.transformer.h[layer].attn.c_proj for layer in layers]
    before = [len(module._forward_hooks) for module in modules]

    injector.attach()
    attached = [len(module._forward_hooks) for module in modules]
    assert attached == [count + 1 for count in before]

    injector.detach()
    detached = [len(module._forward_hooks) for module in modules]
    assert detached == before
