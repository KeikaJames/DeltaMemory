"""A2 — SCAR diagnostics signals.

Confirms SCARInjector emits per-layer diagnostics (proj_mass / ortho_residue /
alpha_drift) into an active DiagnosticRecorder, parallel to the LOPI / mHC
bank pattern. Also confirms zero-impact when no recorder is active.

Uses tiny Qwen3 (modern ``model.layers`` arch) so DiagnosticRecorder's layer
locator finds the decoder blocks. GPT-2's ``transformer.h`` is not yet covered
by ``DiagnosticRecorder._find_decoder_layers`` (tracked separately).
"""
from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from deltamemory.diagnostics import DiagnosticRecorder
from deltamemory.memory.scar_injector import SCARInjector


_VOCAB = 256


def _tiny_qwen3():
    cfg = Qwen3Config(
        vocab_size=_VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).eval()


class _T:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=64):
        ids = [min(ord(ch), _VOCAB - 1) for ch in text][:max_length] or [0]
        x = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": x, "attention_mask": torch.ones_like(x)}


_POS = ["truth alpha", "honest beta", "correct gamma", "faithful delta"]
_NEG = ["false alpha", "lying beta", "wrong gamma", "fake delta"]


def test_scar_emits_proj_signals_when_recorder_active():
    model = _tiny_qwen3()
    tok = _T()
    inj = SCARInjector(model, alpha=0.5, layers=[1], k=2)
    inj.calibrate(_POS, _NEG, tok, max_n=4)

    rec = DiagnosticRecorder(model, patcher=None)
    ids = torch.randint(0, _VOCAB, (1, 16))
    with rec:
        with inj:
            with torch.no_grad():
                model(input_ids=ids)

    df = rec.to_pandas()
    signals = set(df["signal_name"].unique())
    assert "scar_proj_mass" in signals
    assert "scar_ortho_residue" in signals
    assert "scar_alpha_drift" in signals

    # Per-layer scalar contract: token == -1 for SCAR signals
    scar_rows = df[df["signal_name"].str.startswith("scar_")]
    assert (scar_rows["token"] == -1).all()
    assert (scar_rows["layer"] == 1).all()


def test_scar_proj_mass_in_unit_range():
    model = _tiny_qwen3()
    tok = _T()
    inj = SCARInjector(model, alpha=0.5, layers=[1], k=2)
    inj.calibrate(_POS, _NEG, tok, max_n=4)

    rec = DiagnosticRecorder(model, patcher=None)
    ids = torch.randint(0, _VOCAB, (1, 16))
    with rec:
        with inj:
            with torch.no_grad():
                model(input_ids=ids)
    df = rec.to_pandas()
    pm = df[df["signal_name"] == "scar_proj_mass"]["value"].astype(float)
    assert (pm >= 0.0).all() and (pm <= 1.0 + 1e-5).all()
    res = df[df["signal_name"] == "scar_ortho_residue"]["value"].astype(float)
    assert (res >= 0.0).all()


def test_scar_no_emit_when_no_recorder():
    """Sanity: SCAR must not raise / leak when DiagnosticRecorder is unused."""
    model = _tiny_qwen3()
    tok = _T()
    inj = SCARInjector(model, alpha=0.5, layers=[1], k=2)
    inj.calibrate(_POS, _NEG, tok, max_n=4)
    ids = torch.randint(0, _VOCAB, (1, 16))
    with inj:
        with torch.no_grad():
            out = model(input_ids=ids)
    assert torch.isfinite(out.logits).all()


def test_scar_alpha_zero_no_emit():
    """α=0 short-circuits before projection — no diagnostics emitted (matches
    the LOPI silent-when-no-op convention)."""
    model = _tiny_qwen3()
    tok = _T()
    inj = SCARInjector(model, alpha=0.0, layers=[1], k=2)
    inj.calibrate(_POS, _NEG, tok, max_n=4)
    rec = DiagnosticRecorder(model, patcher=None)
    ids = torch.randint(0, _VOCAB, (1, 16))
    with rec:
        with inj:
            with torch.no_grad():
                model(input_ids=ids)
    df = rec.to_pandas()
    if not df.empty:
        signals = set(df["signal_name"].unique())
        assert not any(s.startswith("scar_") for s in signals)
