"""Tests for deltamemory.config pydantic v2 schemas and YAML loader.

Coverage:
- Happy path for each of the 7 injector/service schemas
- top_k > bank_size rejected
- alpha < 0 rejected
- YAML round-trip
- ServiceConfig port out-of-range rejected
- eta_sigma = 0 rejected
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from deltamemory.config.schemas import (
    AttnNativeBankConfig,
    BaseInjectorConfig,
    CaaConfig,
    LopiConfig,
    MnemeWriterConfig,
    RomeWriterConfig,
    ScarConfig,
    ServiceConfig,
)
from deltamemory.config.loader import load_config, dump_config


# ── Happy-path tests (one per schema) ─────────────────────────────────────────

def test_base_injector_config_happy():
    cfg = BaseInjectorConfig(alpha=0.5, layer_idx=4, enabled=True, dtype="fp32", device="cpu")
    assert cfg.alpha == 0.5
    assert cfg.layer_idx == 4
    assert cfg.dtype == "fp32"


def test_lopi_config_happy():
    cfg = LopiConfig(
        alpha=0.5,
        eta_sigma=1.0,
        gate_k=2.0,
        gate_theta=0.1,
        use_derivative_gate=True,
    )
    assert cfg.eta_sigma == 1.0
    assert cfg.use_derivative_gate is True


def test_caa_config_happy():
    cfg = CaaConfig(alpha=0.3, n_pairs=32, use_lopi_gate=False, gate_k=1.5, gate_theta=0.0)
    assert cfg.n_pairs == 32
    assert cfg.use_lopi_gate is False


def test_scar_config_happy():
    cfg = ScarConfig(alpha=1.0, projection="m_perp", subspace_rank=4)
    assert cfg.projection == "m_perp"
    assert cfg.subspace_rank == 4


def test_attn_native_bank_config_happy():
    cfg = AttnNativeBankConfig(
        alpha=0.7,
        bank_size=128,
        top_k=8,
        theta=math.pi / 6,
        capture="pre_rope",
    )
    assert cfg.bank_size == 128
    assert cfg.top_k == 8
    assert cfg.capture == "pre_rope"


def test_rome_writer_config_happy():
    cfg = RomeWriterConfig(alpha=1.0, n_optim_steps=10)
    assert cfg.n_optim_steps == 10


def test_mneme_writer_config_happy():
    cfg = MnemeWriterConfig(alpha=0.9, write_alpha=0.4)
    assert cfg.write_alpha == 0.4


def test_service_config_happy():
    cfg = ServiceConfig(
        bind_host="0.0.0.0",
        port=8080,
        workers=4,
        request_size_cap_mb=16,
    )
    assert cfg.port == 8080
    assert cfg.jwt_required is True  # default
    assert cfg.prometheus_enabled is True  # default


# ── Validation-rejection tests ─────────────────────────────────────────────────

def test_top_k_greater_than_bank_size_rejected():
    with pytest.raises(ValidationError, match="top_k"):
        AttnNativeBankConfig(alpha=0.5, bank_size=4, top_k=8, theta=0.0, capture="pre_rope")


def test_alpha_negative_rejected():
    with pytest.raises(ValidationError):
        LopiConfig(alpha=-0.1, eta_sigma=1.0, gate_k=1.0, gate_theta=0.0)


def test_service_port_out_of_range_rejected():
    with pytest.raises(ValidationError):
        ServiceConfig(bind_host="127.0.0.1", port=99999, workers=1, request_size_cap_mb=1)


def test_eta_sigma_zero_rejected():
    with pytest.raises(ValidationError):
        LopiConfig(alpha=0.5, eta_sigma=0.0, gate_k=1.0, gate_theta=0.0)


# ── YAML round-trip ────────────────────────────────────────────────────────────

def test_yaml_round_trip(tmp_path: Path):
    example_yaml = Path(__file__).parent.parent / "deltamemory" / "config" / "example.yaml"
    loaded = load_config(example_yaml)

    assert "injectors.lopi" in loaded
    assert isinstance(loaded["injectors.lopi"], LopiConfig)
    assert isinstance(loaded["service"], ServiceConfig)

    out_path = tmp_path / "round_trip.yaml"
    dump_config(loaded, out_path)

    reloaded = load_config(out_path)
    assert reloaded["injectors.lopi"].model_dump() == loaded["injectors.lopi"].model_dump()
    assert reloaded["service"].model_dump() == loaded["service"].model_dump()
