"""A5 — Tests for CAA diagnostic signal emission.

CAA injector emits three signals per (step, layer) at token=-1 when a
``DiagnosticRecorder`` is active:

- ``caa_steer_norm``         = α · ‖s‖
- ``caa_gate_mean``          = mean(γ) (1.0 when ungated)
- ``caa_hidden_drift_ratio`` = ‖α·γ·s‖_F / ‖hidden‖_F

Mirrors the SCAR / LOPI emit pattern.  Uses a tiny Qwen3 fixture (rather
than GPT-2) because :class:`DiagnosticRecorder._find_decoder_layers` does
not yet resolve ``transformer.h`` paths; Qwen3's ``model.layers`` is on
the supported list.
"""
from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from deltamemory.diagnostics import DiagnosticRecorder
from deltamemory.memory.caa_injector import CAAConfig, CAAInjector


_VOCAB = 256
_HIDDEN = 64
_LAYERS = 4
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 16


@pytest.fixture(scope="module")
def tiny_qwen3():
    cfg = Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=128,
        num_hidden_layers=_LAYERS,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    return model


def _make_injector(model, alpha: float, *, use_gate: bool = False) -> CAAInjector:
    cfg = CAAConfig(inject_layer=1, alpha=alpha, use_lopi_gate=use_gate)
    inj = CAAInjector(model, cfg)
    g = torch.Generator().manual_seed(13)
    inj.steering_vector = torch.randn(_HIDDEN, generator=g)
    return inj


def _input_ids(seed: int = 42, n: int = 16):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB, (1, n), generator=g)


# ---------------------------------------------------------------------------
# Test 1 — signals emitted with expected names + shape
# ---------------------------------------------------------------------------


def test_caa_diag_signals_emitted(tiny_qwen3):
    model = tiny_qwen3
    inj = _make_injector(model, alpha=1.0)
    ids = _input_ids()

    with DiagnosticRecorder(model, patcher=None) as rec, inj, torch.no_grad():
        model(input_ids=ids, use_cache=False)

    df = rec.to_pandas()
    names = set(df["signal_name"].unique())
    assert {"caa_steer_norm", "caa_gate_mean", "caa_hidden_drift_ratio"}.issubset(
        names
    ), f"missing CAA signals; got {names}"

    sub = df[df["signal_name"].str.startswith("caa_")]
    assert (sub["token"] == -1).all(), "CAA signals must use token=-1"
    assert (sub["layer"] == 1).all(), "all emissions must come from inject_layer=1"


# ---------------------------------------------------------------------------
# Test 2 — value ranges
# ---------------------------------------------------------------------------


def test_caa_diag_value_ranges(tiny_qwen3):
    model = tiny_qwen3
    inj = _make_injector(model, alpha=2.5)

    with DiagnosticRecorder(model, patcher=None) as rec, inj, torch.no_grad():
        model(input_ids=_input_ids(), use_cache=False)

    df = rec.to_pandas()

    norms = df[df["signal_name"] == "caa_steer_norm"]["value"]
    assert (norms > 0).all(), "α·‖s‖ must be positive when α≠0 and s≠0"

    gates = df[df["signal_name"] == "caa_gate_mean"]["value"]
    # ungated → exactly 1.0
    assert (gates == 1.0).all(), f"ungated CAA must report gate_mean=1.0; got {gates.tolist()}"

    drift = df[df["signal_name"] == "caa_hidden_drift_ratio"]["value"]
    assert (drift >= 0).all(), "drift ratio must be non-negative"


# ---------------------------------------------------------------------------
# Test 3 — gated path reports non-trivial gate stats
# ---------------------------------------------------------------------------


def test_caa_diag_gated_path(tiny_qwen3):
    model = tiny_qwen3
    inj = _make_injector(model, alpha=1.0, use_gate=True)

    with DiagnosticRecorder(model, patcher=None) as rec, inj, torch.no_grad():
        model(input_ids=_input_ids(), use_cache=False)

    df = rec.to_pandas()
    gates = df[df["signal_name"] == "caa_gate_mean"]["value"]
    # First forward step has no prev_hidden → gate fallback to 1.0; that's
    # legal — only require the value is in [0, 1].
    assert ((gates >= 0.0) & (gates <= 1.0)).all(), (
        f"gate_mean must be in [0,1]; got {gates.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 4 — no recorder ⇒ no crash, no emit
# ---------------------------------------------------------------------------


def test_caa_diag_no_recorder_safe(tiny_qwen3):
    model = tiny_qwen3
    inj = _make_injector(model, alpha=1.0)

    # No DiagnosticRecorder context → CAA hook must still work and not raise.
    with inj, torch.no_grad():
        out = model(input_ids=_input_ids(), use_cache=False).logits
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Test 5 — α=0 short-circuit ⇒ no CAA signals emitted
# ---------------------------------------------------------------------------


def test_caa_diag_alpha_zero_no_emit(tiny_qwen3):
    model = tiny_qwen3
    inj = _make_injector(model, alpha=0.0)

    with DiagnosticRecorder(model, patcher=None) as rec, inj, torch.no_grad():
        model(input_ids=_input_ids(), use_cache=False)

    df = rec.to_pandas()
    caa_rows = df[df["signal_name"].str.startswith("caa_")]
    assert len(caa_rows) == 0, (
        f"α=0 must short-circuit before diagnostic emit; got {len(caa_rows)} rows"
    )
