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
    """Load gemma-4-E2B for stage-13A gates.

    Skips if the model is not cached locally and offline mode is enforced
    (CI-friendly): we never trigger a multi-GB download from a unit test.
    Device auto-selects CUDA → MPS → CPU; override with ``RCVHC_TEST_DEVICE``.
    """
    from rcvhc.gemma.model_adapter import load_model_bundle
    device = os.environ.get("RCVHC_TEST_DEVICE")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Probe local HF cache before loading.  If we'd have to hit the network
    # and HF_HUB_OFFLINE is set (typical CI), skip rather than fail-loud.
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache(repo_id="google/gemma-4-E2B", filename="config.json")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    if cached is None and offline:
        pytest.skip("google/gemma-4-E2B not in local HF cache and offline mode enforced")

    try:
        bundle = load_model_bundle(
            "google/gemma-4-E2B",
            device=device,
            dtype="bfloat16",
            attn_implementation="eager",
        )
    except Exception as exc:  # network failure, gated repo, OOM, etc.
        pytest.skip(f"could not load gemma-4-E2B: {exc}")
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


def test_state_dict_round_trip(model_bundle):
    """Gate 13A.4: bank.state_dict() ↔ from_state_dict() preserves shape,
    dtype, per-layer head_dims, fact metadata, and read-time behavior."""
    model = model_bundle.model
    tok = model_bundle.tokenizer
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    write_fact(patcher, bank, tok,
               write_prompt="The mayor of Paris is Hidalgo.",
               fact_id="paris_mayor",
               address="mayor of Paris")
    write_fact(patcher, bank, tok,
               write_prompt="The capital of France is Paris.",
               fact_id="france_capital",
               address="capital of France")

    sd = bank.state_dict()
    bank2 = AttnNativeBank.from_state_dict(sd, device=bank.device, dtype=bank.dtype)

    assert bank2.num_layers == bank.num_layers
    assert bank2.num_kv_heads == bank.num_kv_heads
    assert bank2.head_dim == bank.head_dim
    assert bank2.head_dims == bank.head_dims
    assert bank2.fact_ids == bank.fact_ids
    assert bank2.address_strs == bank.address_strs
    for layer in range(bank.num_layers):
        assert bank.M_K[layer].shape == bank2.M_K[layer].shape
        assert bank.M_V[layer].shape == bank2.M_V[layer].shape
        assert bank.M_K[layer].dtype == bank2.M_K[layer].dtype
        assert torch.equal(bank.M_K[layer].cpu(), bank2.M_K[layer].cpu()), \
            f"M_K layer {layer} differs after round-trip"
        assert torch.equal(bank.M_V[layer].cpu(), bank2.M_V[layer].cpu()), \
            f"M_V layer {layer} differs after round-trip"

    prompt = "The current mayor of Paris is"
    out_a = forward_with_bank(patcher, bank,  tok, prompt, alpha=1.0).float()
    out_b = forward_with_bank(patcher, bank2, tok, prompt, alpha=1.0).float()
    diff = (out_a - out_b).abs().max().item()
    print(f"\n[round-trip] post-deserialize logit diff = {diff:.3e}")
    assert diff < 1e-3, f"round-trip changed read-time logits: {diff:.3e}"

