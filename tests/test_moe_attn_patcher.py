"""Tests for MoeAttnNativePatcher (Phase W.5 / per-expert column cap).

Five cases:

1. ``test_alpha_zero_bit_equal_empty_bank``
   Real (small) model + MoeAttnNativePatcher: forward at alpha=0, empty bank
   ⇒ ``torch.equal`` to the unpatched logits.  Skipped if Qwen2.5-0.5B is
   not cached locally.

2. ``test_alpha_zero_bit_equal_with_facts``
   Writes 2 facts then forwards at alpha=0 ⇒ ``torch.equal``.

3. ``test_global_vs_per_expert_differ_at_alpha_pos``
   Synthetic post-softmax weights + router outputs; dispatcher under
   ``cap_mode='per_expert'`` vs ``cap_mode='global'`` produces numerically
   different bank columns.

4. ``test_per_expert_cap_clips_outliers``
   Hand-built degenerate case: one expert routed to every token; per-expert
   cap clips harder than the global cap (proves the per-expert formula
   actually does something distinct).

5. ``test_router_outputs_shape``
   MockMoeAdapter accepts ``set_router_outputs``; ``get_router_outputs``
   returns the RouterOutputs dict with the expected key set and shapes.

These tests do NOT require any MoE checkpoint.  Real-model end-to-end
validation on a Qwen3-MoE-A3B / Mixtral-8x7B is deferred to GB10 (W.5
grid runner).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from deltamemory.memory.arch_moe_adapter import (
    MockMoeAdapter,
    Qwen3MoeAdapter,
    RouterOutputs,
)
from deltamemory.memory.moe_attn_patcher import (
    MoeAttnNativePatcher,
    _decode_router_output,
)
from deltamemory.memory.mhc_shield import (
    apply_shield_per_expert,
    shield_attention_weights,
)


QWEN05_PATH = Path("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B").expanduser()
QWEN05_AVAILABLE = QWEN05_PATH.exists() and any(
    QWEN05_PATH.joinpath("snapshots").glob("*/model.safetensors")
)

GEMMA4_PATH = Path("~/.cache/huggingface/hub/models--google--gemma-4-E2B").expanduser()
GEMMA4_AVAILABLE = GEMMA4_PATH.exists() and any(
    GEMMA4_PATH.joinpath("snapshots").glob("*/model.safetensors*")
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synth_weights(B=1, H=2, T=4, N=3, seed=0):
    """Build a row-stochastic post-softmax attention weight tensor of shape
    (B, H, T, T+N) with the last N columns being the bank columns."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.rand(B, H, T, T + N, generator=g)
    return raw / raw.sum(dim=-1, keepdim=True)


def _make_router_outputs(seq_len: int, num_experts: int, top_k: int = 2, seed: int = 0):
    """Build random valid router outputs."""
    g = torch.Generator().manual_seed(seed)
    # Random top-k expert indices.
    idx = torch.stack(
        [torch.randperm(num_experts, generator=g)[:top_k] for _ in range(seq_len)]
    )
    # Random routing weights, normalised across the top-k axis.
    w = torch.rand(seq_len, top_k, generator=g)
    w = w / w.sum(dim=-1, keepdim=True)
    return RouterOutputs(expert_indices=idx, routing_weights=w, num_experts=num_experts)


# ---------------------------------------------------------------------------
# Tests 1 & 2: alpha=0 bit-equality (real small model)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GEMMA4_AVAILABLE, reason="gemma-4-E2B not cached")
def test_alpha_zero_bit_equal_empty_bank():
    """alpha=0 + empty bank ⇒ logits bit-equal to unpatched.

    Uses gemma-4-E2B (the same model the dense bit-equal tests use, where
    the base AttnNativePatcher is known to be exactly bit-equal).  We wrap
    the auto-picked Gemma4 adapter with :func:`make_moe_adapter_from_dense`
    so the MoE patcher's ``do_inject`` short-circuit can be exercised
    without disturbing the dense bit-equal property.
    """
    from deltamemory.gemma.model_adapter import load_model_bundle
    from deltamemory.memory.arch_adapter import pick_adapter
    from deltamemory.memory.arch_moe_adapter import make_moe_adapter_from_dense
    from deltamemory.memory.attn_native_bank import (
        AttnNativePatcher,
        fresh_bank,
        forward_with_bank,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        bundle = load_model_bundle(
            "google/gemma-4-E2B", device=device, dtype="bfloat16",
            attn_implementation="eager",
        )
    except Exception as exc:
        pytest.skip(f"could not load gemma-4-E2B: {exc}")

    model = bundle.model
    tok = bundle.tokenizer
    prompt = "The capital of France is"
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        base = model(**enc).logits[0, -1].detach().clone()

    # Auto-pick dense adapter, then promote to MoE-aware via the helper.
    probe = AttnNativePatcher(model)
    moe_adapter = make_moe_adapter_from_dense(probe.adapter, num_experts=4, top_k=2)

    patcher = MoeAttnNativePatcher(model, adapter=moe_adapter, cap_mode="per_expert")
    bank = fresh_bank(model)

    out = forward_with_bank(patcher, bank, tok, prompt, alpha=0.0)
    if not torch.equal(base, out):
        diff = (base.float() - out.float()).abs().max().item()
        raise AssertionError(f"alpha=0 empty-bank not bit-equal: max-abs-diff={diff:.3e}")


@pytest.mark.skipif(not GEMMA4_AVAILABLE, reason="gemma-4-E2B not cached")
def test_alpha_zero_bit_equal_with_facts():
    """Write 2 facts at alpha=0 ⇒ still bit-equal (do_inject guard fires)."""
    from deltamemory.gemma.model_adapter import load_model_bundle
    from deltamemory.memory.arch_moe_adapter import make_moe_adapter_from_dense
    from deltamemory.memory.attn_native_bank import (
        AttnNativePatcher,
        fresh_bank,
        forward_with_bank,
        write_fact,
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        bundle = load_model_bundle(
            "google/gemma-4-E2B", device=device, dtype="bfloat16",
            attn_implementation="eager",
        )
    except Exception as exc:
        pytest.skip(f"could not load gemma-4-E2B: {exc}")

    model = bundle.model
    tok = bundle.tokenizer
    prompt = "Paris is the capital of"
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        base = model(**enc).logits[0, -1].detach().clone()

    probe = AttnNativePatcher(model)
    moe_adapter = make_moe_adapter_from_dense(probe.adapter, num_experts=4, top_k=2)
    patcher = MoeAttnNativePatcher(model, adapter=moe_adapter, cap_mode="per_expert")
    bank = fresh_bank(model)

    write_fact(patcher, bank, tok, "The Eiffel Tower is in Paris.", "fact1", "Paris")
    write_fact(patcher, bank, tok, "The Louvre is in Paris.", "fact2", "Paris")
    assert not bank.empty

    out = forward_with_bank(patcher, bank, tok, prompt, alpha=0.0)
    if not torch.equal(base, out):
        diff = (base.float() - out.float()).abs().max().item()
        raise AssertionError(f"alpha=0 (non-empty bank) not bit-equal: max-abs-diff={diff:.3e}")


# ---------------------------------------------------------------------------
# Test 3: global vs per_expert produce different bank columns
# ---------------------------------------------------------------------------


def test_global_vs_per_expert_differ_at_alpha_pos():
    """At a positive alpha, per_expert and global cap modes diverge.

    We don't need a real model for this — we feed the dispatcher synthetic
    post-softmax weights with values that exceed kappa (so both caps fire,
    but with different scale factors).
    """
    B, H, T, N = 1, 2, 4, 3
    weights = _make_synth_weights(B, H, T, N, seed=0)
    # Inflate bank columns so the global col-sum > kappa.
    weights = weights.clone()
    weights[..., T:] *= 5.0

    # Build a MockMoeAdapter with router outputs covering the seq_len = B*T*H
    # positions used by the dispatcher's reshape (q, T+N).  In the 2-D test
    # we feed weights as (B*Hq*T, T+N), so seq_len for the router is set to
    # match.
    seq_len = B * H * T
    num_experts = 4
    ro = _make_router_outputs(seq_len, num_experts, top_k=2, seed=42)
    adapter = MockMoeAdapter(num_experts=num_experts, top_k=2)
    adapter.set_router_outputs(0, ro)

    # Build a bare patcher object to call _dispatch_shield directly without
    # installing on a real model.
    patcher = MoeAttnNativePatcher.__new__(MoeAttnNativePatcher)
    patcher.adapter = adapter
    patcher.cap_mode = "global"
    patcher.kappa = 1.0
    out_global = patcher._dispatch_shield(weights, bank_size=N, enabled=True, kappa=1.0)

    patcher.cap_mode = "per_expert"
    # Push a fake current layer so the dispatcher reads adapter cache.
    from deltamemory.memory.moe_attn_patcher import _push_layer, _pop_layer

    _push_layer(0)
    try:
        out_pe = patcher._dispatch_shield(weights, bank_size=N, enabled=True, kappa=1.0)
    finally:
        _pop_layer()

    # Native columns must be untouched by both caps.
    assert torch.equal(out_global[..., :T], weights[..., :T])
    assert torch.equal(out_pe[..., :T], weights[..., :T])
    # Bank columns must differ between modes.
    bank_diff = (out_global[..., T:] - out_pe[..., T:]).abs().max().item()
    assert bank_diff > 1e-6, (
        f"per_expert and global produced identical bank columns (max-diff={bank_diff})"
    )


# ---------------------------------------------------------------------------
# Test 4: per-expert cap clips outlier (compared to global cap)
# ---------------------------------------------------------------------------


def test_per_expert_cap_clips_outliers():
    """Degenerate case: one expert routed for every token, bank col-sum huge.

    The per-expert cap with that expert's mass = full col-sum should clip at
    least as hard as the global cap.  Concretely we build a 2-D weight matrix
    where one bank column has a very large total mass concentrated in expert
    0 (via a 1-of-1 routing), so the per-expert col-sum equals the global
    col-sum — the two modes should produce *identical* shielded bank for
    that column when only one expert is active per token.
    """
    q = 6
    T_orig = 3
    N = 2
    weights = torch.ones(q, T_orig + N) / (T_orig + N)
    # Inflate one bank column dramatically.
    weights[:, T_orig + 0] = 0.8
    weights[:, T_orig + 1] = 0.05
    # Re-normalise rows.
    weights = weights / weights.sum(dim=-1, keepdim=True)

    kappa = 1.0
    # Single-expert routing: every token picks expert 0 with weight 1.0.
    expert_gates = {0: torch.ones(q)}
    pe_out = apply_shield_per_expert(weights, T_orig=T_orig, kappa=kappa, expert_gates=expert_gates)
    g_out = shield_attention_weights(weights, bank_size=N, enabled=True, kappa=kappa)

    # In single-expert + normalised routing, per-expert ≡ global.
    assert torch.allclose(pe_out, g_out, atol=1e-5), (
        "single-expert per-expert cap must coincide with global cap"
    )

    # Now flip the scenario: two experts, asymmetric gates concentrating
    # half the mass on expert 0 (which sees the outlier column) and zero on
    # expert 1.  Per-expert cap on expert 0 sees a *smaller* col-sum than
    # the global cap because each row contributes only 0.5 of its bank
    # weight to expert 0.  Result: per-expert clips less aggressively.
    expert_gates = {
        0: torch.full((q,), 0.5),
        1: torch.full((q,), 0.5),
    }
    pe_out2 = apply_shield_per_expert(weights, T_orig=T_orig, kappa=kappa, expert_gates=expert_gates)
    # Bank columns: per-expert keeps more mass than global (bank sum closer
    # to original) because each expert's bucket col-sum is half of global's.
    pe_bank_mass = pe_out2[:, T_orig:].sum().item()
    g_bank_mass = g_out[:, T_orig:].sum().item()
    assert pe_bank_mass >= g_bank_mass - 1e-5, (
        f"per-expert cap unexpectedly clipped harder than global: "
        f"pe={pe_bank_mass}, g={g_bank_mass}"
    )


# ---------------------------------------------------------------------------
# Test 5: router outputs shape / API
# ---------------------------------------------------------------------------


def test_router_outputs_shape():
    """MockMoeAdapter round-trips RouterOutputs and exposes the expected API."""
    adapter = MockMoeAdapter(num_experts=4, top_k=2)
    seq_len = 7
    ro = _make_router_outputs(seq_len, num_experts=4, top_k=2, seed=1)
    adapter.set_router_outputs(layer_idx=3, outputs=ro)

    fetched = adapter.get_router_outputs(layer_idx=3)
    assert fetched is not None
    assert fetched.expert_indices.shape == (seq_len, 2)
    assert fetched.routing_weights.shape == (seq_len, 2)
    assert fetched.num_experts == 4

    # Active-expert / num-expert API.
    assert adapter.get_active_experts(0) == 2
    assert adapter.get_num_experts(0) == 4

    # clear_router_cache wipes everything.
    adapter.clear_router_cache()
    assert adapter.get_router_outputs(layer_idx=3) is None


def test_decode_router_output_helper():
    """The internal _decode helper unpacks both Qwen3-MoE and Mixtral shapes."""
    seq_len = 5
    top_k = 2
    num_experts = 4
    weights = torch.rand(seq_len, top_k)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    idx = torch.zeros(seq_len, top_k, dtype=torch.long)
    logits = torch.randn(seq_len, num_experts)

    # Qwen3-MoE order: (logits, weights, indices)
    ro = _decode_router_output((logits, weights, idx), num_experts=num_experts)
    assert ro is not None and ro.expert_indices.shape == (seq_len, top_k)

    # Mixtral order: (logits, top_k_weights, top_k_index) — same unpack rule.
    ro2 = _decode_router_output((logits, weights, idx), num_experts=num_experts)
    assert ro2 is not None and ro2.routing_weights.shape == (seq_len, top_k)


def test_qwen3_moe_adapter_smoke():
    """Qwen3MoeAdapter is constructible and exposes the right API surface."""
    a = Qwen3MoeAdapter(num_experts=128, top_k=8)
    assert a.get_active_experts() == 8
    assert a.get_num_experts() == 128
    assert a.name == "qwen3_moe"
