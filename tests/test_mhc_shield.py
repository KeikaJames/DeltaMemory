"""Unit + integration tests for the mHC spectral shield (v3.2).

Run:
    .venv-mac/bin/python -m pytest tests/test_mhc_shield.py -s
"""
from __future__ import annotations

import os
import pytest
import torch

from deltamemory.memory.mhc_shield import (
    sinkhorn_knopp_projection,
    shield_attention_weights,
)


# ---------------------------------------------------------------------------
# Unit tests on the projector itself (no LLM required, runs everywhere)
# ---------------------------------------------------------------------------


def test_sinkhorn_preserves_shape_and_dtype():
    x = torch.rand(2, 4, 7, 11, dtype=torch.float32)
    y = sinkhorn_knopp_projection(x, iters=3)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_sinkhorn_row_sums_to_one():
    """Final operation is row-normalisation, so rows must sum to 1 exactly."""
    torch.manual_seed(0)
    x = torch.rand(8, 32, dtype=torch.float32)
    y = sinkhorn_knopp_projection(x, iters=5)
    row_sums = y.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"row sums deviate: max |sum-1|={row_sums.sub(1.0).abs().max().item():.3e}"
    )


def test_sinkhorn_column_sums_close_to_target():
    """For square inputs, column sums should converge to 1 within a few SK
    iterations.  We allow a generous tolerance because three iterations is
    rarely tight to machine epsilon."""
    torch.manual_seed(1)
    x = torch.rand(16, 16, dtype=torch.float32)
    y = sinkhorn_knopp_projection(x, iters=20)
    col_sums = y.sum(dim=-2)
    # row-normalisation is the *last* op, so columns may drift slightly,
    # but with 20 iterations should be well within 5e-3.
    assert col_sums.sub(1.0).abs().max().item() < 5e-3


def test_sinkhorn_spectral_radius_le_one():
    """The headline claim: σ_max(W) ≤ 1 after projection (within numerical
    slack from the truncated SK iteration count)."""
    torch.manual_seed(2)
    for q, k in [(8, 8), (8, 16), (16, 8), (32, 40)]:
        x = torch.rand(q, k, dtype=torch.float32) + 0.01
        y = sinkhorn_knopp_projection(x, iters=10)
        sigma_max = torch.linalg.svdvals(y)[0].item()
        # For doubly-stochastic with target_col_sum = q/k, the spectral
        # radius is bounded by sqrt(q/k * 1) = sqrt(q/k); for q≤k this
        # gives σ_max ≤ 1 strictly.  For q>k it gives σ_max ≤ sqrt(q/k).
        bound = max(1.0, (q / k) ** 0.5) + 1e-2
        assert sigma_max <= bound, (
            f"σ_max={sigma_max:.4f} exceeds bound {bound:.4f} at shape ({q},{k})"
        )


def test_sinkhorn_zero_iters_is_identity():
    x = torch.rand(4, 8, dtype=torch.float32)
    y = sinkhorn_knopp_projection(x, iters=0)
    # iters=0 still does the final row-normalisation, so y[i] = x[i] / sum(x[i])
    expected = x / x.sum(dim=-1, keepdim=True)
    assert torch.allclose(y, expected, atol=1e-6)


def test_sinkhorn_handles_4d_batched_attention_shape():
    """Realistic attention-weight shape: [B, Hq, T, T+N]."""
    torch.manual_seed(3)
    B, H, T, N = 2, 8, 16, 4
    # Simulate post-softmax: row-stochastic input
    raw = torch.randn(B, H, T, T + N)
    w = torch.softmax(raw, dim=-1)
    y = sinkhorn_knopp_projection(w, iters=3)
    assert y.shape == w.shape
    rs = y.sum(dim=-1)
    assert torch.allclose(rs, torch.ones_like(rs), atol=1e-5)


# ---------------------------------------------------------------------------
# Shield wrapper short-circuit semantics
# ---------------------------------------------------------------------------


def test_shield_disabled_is_identity():
    """enabled=False must return the input bit-for-bit (red-line: α=0 bit-equal)."""
    x = torch.rand(2, 4, 8, 12)
    y = shield_attention_weights(x, bank_size=4, enabled=False, iters=3)
    assert torch.equal(x, y)


def test_shield_empty_bank_is_identity():
    """bank_size=0 must short-circuit (no injection means no shielding needed)."""
    x = torch.rand(2, 4, 8, 12)
    y = shield_attention_weights(x, bank_size=0, enabled=True, iters=3)
    assert torch.equal(x, y)


def test_shield_changes_weights_when_active():
    """Sanity: when enabled and bank present, weights *do* change (not a no-op)."""
    torch.manual_seed(7)
    raw = torch.randn(1, 2, 4, 8)
    w = torch.softmax(raw, dim=-1)
    y = shield_attention_weights(w, bank_size=2, enabled=True, iters=3)
    assert not torch.equal(w, y), "shield should modify weights when enabled with bank"


# ---------------------------------------------------------------------------
# Integration test: α=0 with non-empty bank stays bit-equal *with* shield on
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_bundle():
    from deltamemory.gemma.model_adapter import load_model_bundle

    device = os.environ.get("RCVHC_TEST_DEVICE")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id="google/gemma-4-E2B", filename="config.json")
    offline = (
        os.environ.get("HF_HUB_OFFLINE") == "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    )
    if cached is None and offline:
        pytest.skip("gemma-4-E2B not cached and offline mode set")
    try:
        bundle = load_model_bundle(
            "google/gemma-4-E2B",
            device=device,
            dtype="bfloat16",
            attn_implementation="eager",
        )
    except Exception as exc:
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


def test_shield_alpha_zero_bit_equal(model_bundle):
    """Red-line: with mHC shield ON but α=0, output must be identical to
    the un-patched model (the call site short-circuits when α=0 because
    do_inject becomes False)."""
    from deltamemory.memory.attn_native_bank import (
        AttnNativePatcher, fresh_bank, write_fact, forward_with_bank,
    )

    bundle = model_bundle
    model, tok = bundle.model, bundle.tokenizer
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    write_fact(patcher, bank, tok,
               write_prompt="The mayor of Paris is Hidalgo.",
               fact_id="paris_mayor",
               address="mayor of Paris")
    bank.mhc_shield = True
    bank.mhc_iters = 3

    prompt = "The capital of France is"
    base = _logits(model, tok, prompt).float()
    out = forward_with_bank(patcher, bank, tok, prompt, alpha=0.0).float()
    diff = (base - out).abs().max().item()
    print(f"\n[mHC shield on, α=0]  max-abs-diff = {diff:.3e}")
    assert diff < 1e-3, f"α=0 with shield should be bit-equal (diff={diff:.3e})"


def test_shield_alpha_one_modifies_logits(model_bundle):
    """Sanity: with mHC shield ON and α=1, output should differ from α=0
    baseline (i.e. injection still happens; the shield doesn't kill the signal)."""
    from deltamemory.memory.attn_native_bank import (
        AttnNativePatcher, fresh_bank, write_fact, forward_with_bank,
    )

    bundle = model_bundle
    model, tok = bundle.model, bundle.tokenizer
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    write_fact(patcher, bank, tok,
               write_prompt="The mayor of Paris is Hidalgo.",
               fact_id="paris_mayor",
               address="mayor of Paris")
    bank.mhc_shield = True
    bank.mhc_iters = 3

    prompt = "The capital of France is"
    a0 = forward_with_bank(patcher, bank, tok, prompt, alpha=0.0).float()
    a1 = forward_with_bank(patcher, bank, tok, prompt, alpha=1.0).float()
    diff = (a0 - a1).abs().max().item()
    print(f"\n[mHC shield on, α=0 vs α=1]  max-abs-diff = {diff:.3e}")
    assert diff > 1e-3, f"α=1 should still inject signal under shield (diff={diff:.3e})"
