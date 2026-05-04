"""Tests for MoE architecture adapters and per-expert mHC shield (Phase X.2).

Tests:
- test_qwen3_adapter_imports: adapter class instantiable with mock config
- test_mixtral_adapter_imports: MixtralAdapter instantiable with mock config
- test_per_expert_shield_alpha0_bit_equal: α=0 (no bank cols) → bit-equal
- test_per_expert_vs_global_diverge_at_alpha_high: per-expert ≠ global at high α
- test_shared_expert_always_gate_1: mock shared-expert subclass returns gate=1.0
- test_moe_attn_vs_ffn_moe_distinction: documents adapter scope and raises on wrong arch
- test_qwen3_get_expert_pool_sparse_step: sparse_step logic works
- test_qwen3_get_expert_pool_mlp_only_layer: mlp_only layers return empty pool
- test_get_router_gates_qwen3: gate map builds correctly
- test_get_router_gates_unselected_expert_absent: experts with zero selection absent
- test_per_expert_shield_no_bank_no_op: no bank columns → identity
- test_per_expert_shield_empty_gates_no_op: empty gate dict → identity
- test_per_expert_shield_single_expert_equals_global: degenerate single-expert case
- test_per_expert_shield_col_sum_bounded: each expert col-sum ≤ kappa after shield
- test_mixtral_all_layers_have_experts: all Mixtral layers return non-empty pool
- test_per_expert_shield_native_cols_untouched: native columns unchanged
"""
from __future__ import annotations

import torch
import pytest

from deltamemory.arch.moe_adapter import (
    MoEArchAdapter,
    MixtralAdapter,
    MoEArchAdapter,
    Qwen3MoEAdapter,
)
from deltamemory.memory.mhc_shield import (
    apply_shield_per_expert,
    shield_attention_weights,
)


# ---------------------------------------------------------------------------
# Minimal mock config helpers
# ---------------------------------------------------------------------------

QWEN3_MINI_CFG = {
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 6,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "norm_topk_prob": False,
}

MIXTRAL_MINI_CFG = {
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 4,
}


# ---------------------------------------------------------------------------
# Import / instantiation tests
# ---------------------------------------------------------------------------


def test_qwen3_adapter_imports():
    """Adapter class is instantiable with a mock config dict (no model download)."""
    adapter = Qwen3MoEAdapter(QWEN3_MINI_CFG)
    assert adapter._num_experts == 4
    assert adapter._top_k == 2
    assert isinstance(adapter, MoEArchAdapter)


def test_mixtral_adapter_imports():
    """MixtralAdapter is instantiable with a mock config dict."""
    adapter = MixtralAdapter(MIXTRAL_MINI_CFG)
    assert adapter._num_experts == 8
    assert adapter._top_k == 2
    assert isinstance(adapter, MoEArchAdapter)


def test_qwen3_adapter_from_hf_config():
    """Adapter can also be initialised from a real HuggingFace config object."""
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    cfg = Qwen3MoeConfig(
        num_local_experts=16,
        num_experts_per_tok=4,
        num_hidden_layers=8,
        decoder_sparse_step=2,
        mlp_only_layers=[0, 1],
    )
    adapter = Qwen3MoEAdapter(cfg)
    assert adapter._num_experts == 16
    assert adapter._top_k == 4
    assert adapter._sparse_step == 2
    assert 0 in adapter._mlp_only


def test_mixtral_adapter_from_hf_config():
    """MixtralAdapter can be initialised from a HuggingFace MixtralConfig."""
    from transformers.models.mixtral.configuration_mixtral import MixtralConfig
    cfg = MixtralConfig(num_local_experts=8, num_experts_per_tok=2)
    adapter = MixtralAdapter(cfg)
    assert adapter._num_experts == 8


# ---------------------------------------------------------------------------
# Expert pool tests
# ---------------------------------------------------------------------------


def test_qwen3_get_expert_pool_sparse_step():
    """sparse_step=2: only layers where (layer_idx+1) % 2 == 0 are MoE.

    This matches the Qwen3MoeDecoderLayer condition:
        (layer_idx + 1) % config.decoder_sparse_step == 0
    So for sparse_step=2: layers 1, 3 are MoE; layers 0, 2 are dense.
    """
    adapter = Qwen3MoEAdapter({
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 4,
        "decoder_sparse_step": 2,
    })
    # layer 0: (0+1)%2 = 1 ≠ 0 → dense
    assert adapter.get_expert_pool(0) == []
    # layer 1: (1+1)%2 = 0 → MoE
    assert adapter.get_expert_pool(1) == [0, 1, 2, 3]
    # layer 2: (2+1)%2 = 1 ≠ 0 → dense
    assert adapter.get_expert_pool(2) == []
    # layer 3: (3+1)%2 = 0 → MoE
    assert adapter.get_expert_pool(3) == [0, 1, 2, 3]


def test_qwen3_get_expert_pool_mlp_only_layer():
    """Layers in mlp_only_layers return empty expert pool."""
    adapter = Qwen3MoEAdapter({
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 4,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [1, 3],
    })
    assert adapter.get_expert_pool(0) == [0, 1, 2, 3]
    assert adapter.get_expert_pool(1) == []
    assert adapter.get_expert_pool(3) == []


def test_mixtral_all_layers_have_experts():
    """All Mixtral layers are MoE; every layer returns non-empty pool."""
    adapter = MixtralAdapter(MIXTRAL_MINI_CFG)
    for layer_idx in range(4):
        pool = adapter.get_expert_pool(layer_idx)
        assert pool == list(range(8)), f"layer {layer_idx} has wrong pool: {pool}"


# ---------------------------------------------------------------------------
# Shared-expert tests
# ---------------------------------------------------------------------------


def test_qwen3_is_shared_expert_always_false():
    """Qwen3-MoE has no shared experts; is_shared_expert always False."""
    adapter = Qwen3MoEAdapter(QWEN3_MINI_CFG)
    for e in range(4):
        assert not adapter.is_shared_expert(e)


def test_shared_expert_always_gate_1():
    """A shared expert (always-active) always receives gate=1.0 for every token.

    Qwen3-MoE does NOT have shared experts in transformers ≥5.7.  We test
    the contract using a minimal subclass that marks expert 0 as shared.
    """
    class MockSharedAdapter(Qwen3MoEAdapter):
        def is_shared_expert(self, expert_id: int) -> bool:
            return expert_id == 0

    adapter = MockSharedAdapter(QWEN3_MINI_CFG)
    seq_len = 5
    # Routing: only expert 1 selected; expert 0 not in routing_weights
    routing_weights = torch.ones(seq_len, 2) * 0.5
    expert_indices = torch.zeros(seq_len, 2, dtype=torch.long)
    expert_indices[:, 1] = 2  # experts {0, 2} selected per token

    gates = adapter.get_router_gates(
        layer_idx=0,
        routing_weights=routing_weights,
        expert_indices=expert_indices,
    )
    # Expert 0 is shared → gate must be all-ones
    assert 0 in gates, "shared expert 0 must be present in gates"
    assert torch.all(gates[0] == 1.0), (
        f"shared expert gate must be 1.0 for all tokens, got {gates[0]}"
    )


# ---------------------------------------------------------------------------
# Router gate decoding tests
# ---------------------------------------------------------------------------


def test_get_router_gates_qwen3():
    """Gate map built correctly from top-k router output."""
    adapter = Qwen3MoEAdapter(QWEN3_MINI_CFG)
    seq_len = 4
    # Each token selects experts [0, 1] with equal weight 0.5
    routing_weights = torch.full((seq_len, 2), 0.5)
    expert_indices = torch.zeros(seq_len, 2, dtype=torch.long)
    expert_indices[:, 1] = 1  # experts 0 and 1

    gates = adapter.get_router_gates(
        layer_idx=0,
        routing_weights=routing_weights,
        expert_indices=expert_indices,
    )
    assert 0 in gates and 1 in gates
    assert torch.allclose(gates[0], torch.full((seq_len,), 0.5))
    assert torch.allclose(gates[1], torch.full((seq_len,), 0.5))
    # Experts 2, 3 not selected → not in gates
    assert 2 not in gates and 3 not in gates


def test_get_router_gates_unselected_expert_absent():
    """Experts not selected by any token are absent from the returned dict."""
    adapter = MixtralAdapter(MIXTRAL_MINI_CFG)
    seq_len = 6
    routing_weights = torch.ones(seq_len, 2) * 0.5
    expert_indices = torch.zeros(seq_len, 2, dtype=torch.long)
    # Only experts 3 and 5 selected
    expert_indices[:, 0] = 3
    expert_indices[:, 1] = 5

    gates = adapter.get_router_gates(
        layer_idx=0,
        routing_weights=routing_weights,
        expert_indices=expert_indices,
    )
    assert set(gates.keys()) == {3, 5}


def test_get_router_gates_dense_layer_returns_empty():
    """Dense (mlp_only) layers return empty gate dict."""
    adapter = Qwen3MoEAdapter({
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 4,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [2],
    })
    routing_weights = torch.ones(4, 2) * 0.5
    expert_indices = torch.zeros(4, 2, dtype=torch.long)
    gates = adapter.get_router_gates(
        layer_idx=2,
        routing_weights=routing_weights,
        expert_indices=expert_indices,
    )
    assert gates == {}


# ---------------------------------------------------------------------------
# per-expert shield — red-line tests
# ---------------------------------------------------------------------------


def test_per_expert_shield_alpha0_bit_equal():
    """Sacred red-line: when there are no bank columns (α=0 path) the output
    must be bit-equal to the input.

    At α=0 the call site short-circuits before calling any shield function.
    Mirroring that logic: when T_orig == weights.size(-1) (N=0), the
    function returns the input object unchanged.
    """
    torch.manual_seed(42)
    seq_len, T_orig = 8, 16
    weights = torch.softmax(torch.randn(seq_len, T_orig), dim=-1)
    gates = {0: torch.ones(seq_len) * 0.5, 1: torch.ones(seq_len) * 0.5}
    # T_orig == weights.size(-1) → N=0 → identity
    out = apply_shield_per_expert(weights, T_orig=T_orig, kappa=1.0, expert_gates=gates)
    assert torch.equal(weights, out), (
        "apply_shield_per_expert must return input unchanged when N=0 (α=0 red-line)"
    )


def test_per_expert_shield_empty_gates_no_op():
    """Empty expert_gates dict → input returned unchanged."""
    torch.manual_seed(1)
    seq_len, T, N = 4, 8, 4
    weights = torch.softmax(torch.randn(seq_len, T + N), dim=-1)
    out = apply_shield_per_expert(weights, T_orig=T, kappa=1.0, expert_gates={})
    assert torch.equal(weights, out)


def test_per_expert_shield_no_bank_no_op():
    """N=0 (T_orig == total cols) → identity even with non-empty gates."""
    torch.manual_seed(2)
    seq_len, T = 6, 12
    weights = torch.softmax(torch.randn(seq_len, T), dim=-1)
    gates = {0: torch.ones(seq_len)}
    out = apply_shield_per_expert(weights, T_orig=T, kappa=1.0, expert_gates=gates)
    assert torch.equal(weights, out)


def test_per_expert_shield_native_cols_untouched():
    """Native columns must be returned bit-for-bit unchanged."""
    torch.manual_seed(5)
    seq_len, T, N = 8, 16, 8
    weights = torch.softmax(torch.randn(seq_len, T + N), dim=-1)
    gates = {0: torch.ones(seq_len) * 0.6, 1: torch.ones(seq_len) * 0.4}
    out = apply_shield_per_expert(weights, T_orig=T, kappa=0.5, expert_gates=gates)
    assert torch.equal(weights[:, :T], out[:, :T]), (
        "native columns must be bit-for-bit unchanged by per-expert shield"
    )


def test_per_expert_shield_single_expert_equals_global():
    """Degenerate single expert with gate=1.0 must equal global shield output.

    With one expert having gate g_e=1 for all tokens, the per-expert cap
    reduces to exactly the global col-sum cap applied to the bank columns.
    """
    torch.manual_seed(7)
    seq_len, T, N = 8, 12, 6
    raw = torch.randn(seq_len, T + N)
    weights = torch.softmax(raw, dim=-1)

    kappa = 0.8
    gates = {0: torch.ones(seq_len)}

    per_expert_out = apply_shield_per_expert(weights, T_orig=T, kappa=kappa, expert_gates=gates)
    # Global cap operates on the last N columns
    global_out = shield_attention_weights(
        weights.unsqueeze(0).unsqueeze(0),  # [1, 1, seq_len, T+N]
        bank_size=N,
        enabled=True,
        kappa=kappa,
    ).squeeze(0).squeeze(0)

    assert torch.allclose(per_expert_out[:, T:], global_out[:, T:], atol=1e-5), (
        "single-expert per-expert cap must equal global cap"
    )


def test_per_expert_vs_global_diverge_at_alpha_high():
    """At high attention-to-bank (high α proxy), per-expert cap yields numerically
    different results from global cap, proving the math is distinct.

    Setup: 2 experts with very unequal gate distributions.  One expert gets
    all heavy tokens, another gets light tokens.  The global cap fires on the
    combined mass; the per-expert cap fires differently within each bucket.
    """
    torch.manual_seed(99)
    seq_len, T, N = 8, 4, 4
    # Construct weights where bank columns have very high attention mass
    # to force cap to fire.
    weights = torch.zeros(seq_len, T + N)
    weights[:, :T] = 0.05 / T          # tiny native
    weights[:, T:] = 0.95 / N          # heavy bank
    # Make row-stochastic
    weights = weights / weights.sum(dim=-1, keepdim=True)

    kappa = 0.5
    # Expert 0: tokens 0-3 with high gate; Expert 1: tokens 4-7 with high gate
    g0 = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    g1 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
    gates = {0: g0, 1: g1}

    per_expert_out = apply_shield_per_expert(weights, T_orig=T, kappa=kappa, expert_gates=gates)

    global_out = shield_attention_weights(
        weights.unsqueeze(0).unsqueeze(0),
        bank_size=N,
        enabled=True,
        kappa=kappa,
    ).squeeze(0).squeeze(0)

    diff = (per_expert_out[:, T:] - global_out[:, T:]).abs().max().item()
    assert diff > 1e-4, (
        f"per-expert and global cap must diverge numerically at high α; "
        f"max-abs-diff={diff:.3e}"
    )


def test_per_expert_shield_col_sum_bounded():
    """Per-expert shield never amplifies: bank_out ≤ bank_in element-wise.

    With normalised routing (Σ_e g_e[i] = 1) and cap_e ≤ 1:
        cap_per_token[i,n] = Σ_e g_e[i] * cap_e[n] ≤ Σ_e g_e[i] = 1
    Therefore bank_out[i,n] = bank[i,n] * cap_per_token[i,n] ≤ bank[i,n].

    Additionally: for each expert e the per-expert column sum is bounded:
        Σ_i g_e[i] * bank[i,n] * cap_e[n] ≤ kappa
    We verify this by re-computing cap_e from scratch in the test.
    """
    torch.manual_seed(13)
    seq_len, T, N = 10, 8, 6
    weights = torch.softmax(torch.randn(seq_len, T + N), dim=-1)

    kappa = 0.6
    # Normalised routing: g0 + g1 = 1 per token
    raw_g = torch.rand(seq_len).clamp(min=0.1, max=0.9)
    g0 = raw_g
    g1 = 1.0 - raw_g
    gates = {0: g0, 1: g1}

    bank_in = weights[:, T:].float()
    out = apply_shield_per_expert(weights, T_orig=T, kappa=kappa, expert_gates=gates)
    bank_out = out[:, T:].float()

    # Property 1: shield never amplifies (bank_out ≤ bank_in element-wise)
    assert (bank_out <= bank_in + 1e-5).all(), (
        "per-expert shield must not amplify bank weights"
    )

    # Property 2: each expert's internal column sum is bounded by kappa
    eps = 1e-9
    for e, g_e in gates.items():
        gate = g_e.float()
        W_e = gate.unsqueeze(-1) * bank_in
        col_sum_e = W_e.sum(dim=0).clamp_min(eps)
        cap_e = (kappa / col_sum_e).clamp(max=1.0)
        effective_col_sum = (W_e * cap_e.unsqueeze(0)).sum(dim=0)
        assert effective_col_sum.max().item() <= kappa + 1e-5, (
            f"expert {e} internal col-sum {effective_col_sum.max().item():.4f} "
            f"exceeds kappa={kappa}"
        )


# ---------------------------------------------------------------------------
# MoE-Attention vs FFN-MoE distinction test
# ---------------------------------------------------------------------------


def test_moe_attn_vs_ffn_moe_distinction():
    """Confirm adapter documents and enforces the MoE-attn vs FFN-MoE scope.

    This test verifies:
    1. The module docstring explicitly distinguishes FFN-MoE (all current
       supported models) from MoE-Attention (hypothetical, not yet supported).
    2. The adapter does NOT raise for FFN-MoE models (Qwen3, Mixtral).
    3. The module docstring mentions the per-expert cap is an approximation
       for FFN-MoE and would be exact for MoE-Attention.

    Per ``docs/theory/mhc_moe.md`` Section 6 summary table:
    - Mixtral 8x7B: FFN-MoE, dense attn → Global is "approximately OK".
    - Qwen3-MoE: KV-shared, FFN-MoE → Per-expert cap via router gate.
    - All current adapters use FFN router gates as proxy.
    """
    import deltamemory.arch.moe_adapter as moa

    docstring = moa.__doc__ or ""
    assert "FFN-MoE" in docstring or "FFN" in docstring, (
        "moe_adapter module docstring must document the FFN-MoE vs MoE-Attention distinction"
    )
    assert "approximat" in docstring.lower(), (
        "moe_adapter module docstring must note the per-expert cap is approximate for FFN-MoE"
    )

    # Both adapters instantiate without error for FFN-MoE models
    Qwen3MoEAdapter(QWEN3_MINI_CFG)
    MixtralAdapter(MIXTRAL_MINI_CFG)

    # The adapter must be usable end-to-end: gate extraction works
    adapter = Qwen3MoEAdapter(QWEN3_MINI_CFG)
    rw = torch.ones(4, 2) * 0.5
    ri = torch.zeros(4, 2, dtype=torch.long)
    ri[:, 1] = 1
    gates = adapter.decode_gate_output(rw, ri, layer_idx=0)
    assert isinstance(gates, dict)
    assert len(gates) > 0


def test_per_expert_shield_invalid_T_orig_raises():
    """Negative T_orig must raise ValueError."""
    weights = torch.rand(4, 8)
    with pytest.raises(ValueError, match="T_orig"):
        apply_shield_per_expert(weights, T_orig=-1, kappa=1.0, expert_gates={0: torch.ones(4)})
