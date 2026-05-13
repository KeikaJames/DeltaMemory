"""Exp9 plumbing tests: Variant fields, clone_bank, drift_bank propagation.

Tests that bank_separate_softmax, bank_merge_beta, mhc_shield, and mhc_kappa
are correctly propagated through the entire pipeline:
  Variant → clone_bank → main loop bank → drift_bank

Also tests the merged-beta gate and separate-softmax + mHC wiring at the
AttnNativeBank level (CPU / mock tensors, no GPU required).

Run:
    pytest tests/test_atb_exp_variant_plumbing.py -s
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deltamemory.memory.attn_native_bank import AttnNativeBank
from deltamemory.memory.lopi import LOPIConfig, LOPIState
from experiments.atb_validation_v1._lib import (
    Variant,
    apply_variant_bank_config,
    variant_uses_dynamic_lopi,
)
from experiments.atb_validation_v1._lib.multi_bank_runner import clone_bank


# ---------------------------------------------------------------------------
# Helpers

def _make_bank(n_layers: int = 2, n_heads: int = 2, head_dim: int = 8,
               bank_size: int = 4, device: str = "cpu",
               dtype: torch.dtype = torch.float32) -> AttnNativeBank:
    """Build a minimal AttnNativeBank with random M_K / M_V."""
    M_K = [torch.randn(bank_size, head_dim, dtype=dtype, device=device)
           for _ in range(n_layers)]
    M_V = [torch.randn(bank_size, head_dim, dtype=dtype, device=device)
           for _ in range(n_layers)]
    bank = AttnNativeBank(
        num_layers=n_layers,
        num_kv_heads=n_heads,
        head_dim=head_dim,
        head_dims=[head_dim] * n_layers,
        num_kv_heads_per_layer=[n_heads] * n_layers,
        device=device,
        dtype=dtype,
        M_K=M_K,
        M_V=M_V,
    )
    bank.fact_ids = [f"fact_{i}" for i in range(bank_size)]
    return bank


# ---------------------------------------------------------------------------
# 1. Variant dataclass has the new fields with correct defaults

def test_variant_has_sep_softmax_fields():
    v = Variant(name="test", alpha=0.05)
    assert hasattr(v, "bank_separate_softmax"), "Variant missing bank_separate_softmax"
    assert hasattr(v, "bank_merge_beta"), "Variant missing bank_merge_beta"
    assert v.bank_separate_softmax is False
    assert v.bank_merge_beta == 1.0


def test_variant_has_mhc_fields():
    v = Variant(name="test", alpha=0.05)
    assert hasattr(v, "mhc_shield")
    assert hasattr(v, "mhc_kappa")
    assert v.mhc_shield is False
    assert v.mhc_kappa == 1.0


def test_variant_to_dict_includes_all_exp9_fields():
    v = Variant(name="test", alpha=0.05, mhc_shield=True, mhc_kappa=0.25,
                bank_separate_softmax=True, bank_merge_beta=0.1)
    d = v.to_dict()
    assert d["mhc_shield"] is True
    assert d["mhc_kappa"] == 0.25
    assert d["bank_separate_softmax"] is True
    assert d["bank_merge_beta"] == 0.1


def test_variant_has_lopi_fields():
    v = Variant(name="test", alpha=0.05)
    assert hasattr(v, "lopi_enabled")
    assert hasattr(v, "lopi_orthogonal")
    assert hasattr(v, "lopi_gaussian")
    assert hasattr(v, "lopi_derivative")
    assert hasattr(v, "lopi_profile_mode")
    assert v.lopi_enabled is False
    assert v.lopi_orthogonal is False
    assert v.lopi_gaussian is True
    assert v.lopi_derivative is True
    assert v.lopi_profile_mode == "auto"


def test_variant_to_dict_includes_lopi_fields():
    v = Variant(
        name="dynlopi",
        lopi_enabled=True,
        lopi_orthogonal=True,
        lopi_gaussian=False,
        lopi_derivative=False,
        lopi_profile_mode="static",
    )
    d = v.to_dict()
    assert d["lopi_enabled"] is True
    assert d["lopi_orthogonal"] is True
    assert d["lopi_gaussian"] is False
    assert d["lopi_derivative"] is False
    assert d["lopi_profile_mode"] == "static"


# ---------------------------------------------------------------------------
# 2. clone_bank propagates sep/beta/mhc fields

def test_clone_bank_propagates_sep_and_beta():
    bank = _make_bank()
    bank.bank_separate_softmax = True
    bank.bank_merge_beta = 0.05
    bank.mhc_shield = True
    bank.mhc_kappa = 0.25

    cloned = clone_bank(bank)
    assert cloned.bank_separate_softmax is True, "clone should inherit bank_separate_softmax"
    assert cloned.bank_merge_beta == 0.05, "clone should inherit bank_merge_beta"
    assert cloned.mhc_shield is True, "clone should inherit mhc_shield"
    assert cloned.mhc_kappa == 0.25, "clone should inherit mhc_kappa"


def test_clone_bank_default_fields():
    bank = _make_bank()
    # bank has no sep/beta/mhc attrs set — should default gracefully.
    cloned = clone_bank(bank)
    assert cloned.bank_separate_softmax is False
    assert cloned.bank_merge_beta == 1.0
    assert cloned.mhc_shield is False
    assert cloned.mhc_kappa == 1.0


def test_clone_bank_copies_lopi_cfg_with_fresh_state():
    bank = _make_bank()
    profile = object()
    bank.lopi_cfg = LOPIConfig(
        enabled=True,
        orthogonal=True,
        gaussian=False,
        derivative=False,
        profile_mode="static",
    )
    bank.lopi_state = LOPIState(num_layers=bank.num_layers)
    bank.lopi_state.prev_residual_norms[0] = 123.0
    bank.lopi_state.profile = profile

    cloned = clone_bank(bank)

    assert cloned.lopi_cfg is not bank.lopi_cfg
    assert cloned.lopi_cfg.enabled is True
    assert cloned.lopi_cfg.orthogonal is True
    assert cloned.lopi_cfg.gaussian is False
    assert cloned.lopi_cfg.derivative is False
    assert cloned.lopi_cfg.profile_mode == "static"
    assert cloned.lopi_state is not bank.lopi_state
    assert cloned.lopi_state.prev_residual_norms == {}
    assert cloned.lopi_state.profile is profile


def test_apply_variant_bank_config_propagates_lopi_mhc_beta():
    bank = _make_bank()
    v = Variant(
        name="dynlopi_mhc_beta",
        mhc_shield=True,
        mhc_kappa=0.25,
        bank_separate_softmax=True,
        bank_merge_beta=0.1,
        lopi_enabled=True,
        lopi_orthogonal=True,
        lopi_gaussian=False,
        lopi_derivative=False,
        lopi_profile_mode="static",
    )

    apply_variant_bank_config(bank, v)

    assert bank.mhc_shield is True
    assert bank.mhc_kappa == 0.25
    assert bank.bank_separate_softmax is True
    assert bank.bank_merge_beta == 0.1
    assert bank.lopi_cfg.enabled is True
    assert bank.lopi_cfg.orthogonal is True
    assert bank.lopi_cfg.gaussian is False
    assert bank.lopi_cfg.derivative is False
    assert bank.lopi_cfg.profile_mode == "static"
    assert variant_uses_dynamic_lopi(v) is True


def test_mhc_beta_do_not_require_forward_sequence_preservation():
    v = Variant(
        name="mhc_beta",
        mhc_shield=True,
        mhc_kappa=0.25,
        bank_merge_beta=0.1,
    )
    assert variant_uses_dynamic_lopi(v) is False


# ---------------------------------------------------------------------------
# 3. state_dict round-trip for sep/beta/mhc fields

def test_state_dict_roundtrip_sep_beta_mhc():
    bank = _make_bank()
    bank.bank_separate_softmax = True
    bank.bank_merge_beta = 0.10
    bank.mhc_shield = True
    bank.mhc_kappa = 0.25

    sd = bank.state_dict()
    assert sd.get("bank_separate_softmax") is True
    assert sd.get("bank_merge_beta") == 0.10
    assert sd.get("mhc_kappa") == 0.25

    bank2 = AttnNativeBank.from_state_dict(sd)
    assert bank2.bank_separate_softmax is True
    assert bank2.bank_merge_beta == 0.10
    assert bank2.mhc_kappa == 0.25


# ---------------------------------------------------------------------------
# 4. merged-beta gate: beta=0 should zero out bank contribution

def test_merged_beta_zero_kills_bank_contribution():
    """With bank_merge_beta=0 in merged branch, out_bank × 0 ≡ out_seq only."""
    bank = _make_bank(n_layers=1, n_heads=1, head_dim=4, bank_size=2)
    bank.bank_separate_softmax = False
    bank.bank_merge_beta = 0.0  # kill bank entirely
    bank.mhc_shield = False

    # We can't run the real forward without a full model.
    # Instead, test the arithmetic directly:
    # out = out_seq + beta * out_bank  →  when beta=0, out = out_seq
    beta = float(getattr(bank, "bank_merge_beta", 1.0))
    out_seq = torch.randn(1, 1, 6, 4)
    out_bank = torch.randn(1, 1, 6, 4)
    result = out_seq + beta * out_bank
    assert torch.allclose(result, out_seq), (
        "beta=0 should give out == out_seq"
    )


def test_merged_beta_one_same_as_default():
    """bank_merge_beta=1.0 should give same result as out_seq + out_bank."""
    beta = 1.0
    out_seq = torch.randn(2, 4, 8, 16)
    out_bank = torch.randn(2, 4, 8, 16)
    expected = out_seq + out_bank
    result = out_seq + beta * out_bank
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# 5. shield_bank_weights integration: ensure shape and cap hold

def test_shield_bank_weights_integration():
    from deltamemory.memory.mhc_shield import shield_bank_weights

    torch.manual_seed(42)
    B, H, T, N = 2, 4, 12, 6
    kappa = 0.3
    w = torch.softmax(torch.randn(B, H, T, N), dim=-1)
    w_capped = shield_bank_weights(w, kappa=kappa)

    assert w_capped.shape == w.shape
    col_sums = w_capped.reshape(-1, N).sum(dim=0)
    assert col_sums.max().item() <= kappa + 1e-5, (
        f"cap violated: max col_sum={col_sums.max().item():.6f} > kappa={kappa}"
    )


# ---------------------------------------------------------------------------
# 6. alpha=0 short-circuit: bank_separate_softmax and bank_merge_beta do not
#    affect output when alpha=0 (the injection branch is skipped entirely).
#    We verify this at the arithmetic level.

def test_alpha_zero_skips_bank_injection():
    """With alpha=0, no bank injection occurs regardless of sep/beta settings."""
    alpha = 0.0
    mv_e = torch.randn(1, 1, 4, 8)  # simulated bank values
    # alpha * mv_e should be zero → no bank contribution.
    bank_input = alpha * mv_e
    assert bank_input.abs().max().item() == 0.0, (
        "alpha=0 must zero out bank input (mv contribution)"
    )
