"""Smoke / unit tests for AttentionNative DeltaMemory bank.

Stage 13A gate 1: with empty bank or alpha=0, the patched model output must
be bit-identical to the unpatched model.

Stage 13A gate 2: writing a fact and reading it back at alpha=1 should bias
the next-token logit toward the value token.

These tests run on whatever device the model is loaded on.  On Apple Silicon
they default to MPS bf16; on GB10 they will run on CUDA bf16.

Run:
    .venv-mac/bin/python -m pytest tests/test_attn_native_bank.py -s
"""
from __future__ import annotations

import os
import pytest
import torch

from rcvhc.memory.attn_native_bank import (
    AttnNativeBank, AttnNativePatcher, fresh_bank, write_fact, forward_with_bank,
)


@pytest.fixture(scope="module")
def model_bundle():
    from rcvhc.gemma.model_adapter import load_model_bundle
    bundle = load_model_bundle(
        "google/gemma-4-E2B",
        device="mps",
        dtype="bfloat16",
        attn_implementation="eager",
    )
    return bundle


def _logits(model, tokenizer, prompt: str) -> torch.Tensor:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
    last = am.sum(dim=1).item() - 1
    return out.logits[0, last].detach()


def test_empty_bank_is_bit_equal(model_bundle):
    """Gate 13A.1: empty bank => patched output == baseline output."""
    model = model_bundle.model
    tok = model_bundle.tokenizer
    prompt = "The capital of France is"
    base = _logits(model, tok, prompt).float()
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    patched = forward_with_bank(patcher, bank, tok, prompt, alpha=1.0).float()
    diff = (base - patched).abs().max().item()
    print(f"\n[bit-equal empty]  max-abs-diff = {diff:.3e}")
    assert diff < 1e-3, f"empty-bank deviation too large: {diff:.3e}"


def test_alpha_zero_is_bit_equal_with_facts(model_bundle):
    """Gate 13A.2: alpha=0 with non-empty bank => still bit-equal."""
    model = model_bundle.model
    tok = model_bundle.tokenizer
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    write_fact(patcher, bank, tok,
               write_prompt="The mayor of Paris is Hidalgo.",
               fact_id="paris_mayor",
               address="mayor of Paris")
    prompt = "The capital of France is"
    base = _logits(model, tok, prompt).float()
    out = forward_with_bank(patcher, bank, tok, prompt, alpha=0.0).float()
    diff = (base - out).abs().max().item()
    print(f"\n[alpha=0 with bank]  max-abs-diff = {diff:.3e}")
    assert diff < 1e-3, f"alpha=0 deviation too large: {diff:.3e}"


def test_single_fact_recall(model_bundle):
    """Gate 13A.3: write one fact; alpha=1 should boost the value token."""
    model = model_bundle.model
    tok = model_bundle.tokenizer
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)

    write_fact(patcher, bank, tok,
               write_prompt="The mayor of Paris is Hidalgo.",
               fact_id="paris_mayor",
               address="mayor of Paris")

    prompt = "The current mayor of Paris is"
    base_logits = _logits(model, tok, prompt).float()
    inj_logits  = forward_with_bank(patcher, bank, tok, prompt, alpha=1.0).float()

    # Locate "Hidalgo" token id (first sub-token of " Hidalgo")
    cand_ids = tok.encode(" Hidalgo", add_special_tokens=False)
    target = cand_ids[0]

    base_rank = (base_logits.argsort(descending=True) == target).nonzero(as_tuple=True)[0].item()
    inj_rank  = (inj_logits.argsort(descending=True) == target).nonzero(as_tuple=True)[0].item()
    base_logit = base_logits[target].item()
    inj_logit  = inj_logits[target].item()

    print(f"\n[recall] target ' Hidalgo' tok={target}  "
          f"base_rank={base_rank}  inj_rank={inj_rank}  "
          f"base_logit={base_logit:.2f}  inj_logit={inj_logit:.2f}")
    assert inj_rank <= base_rank, "injection did not improve target rank"
    assert inj_logit > base_logit, "injection did not raise target logit"
