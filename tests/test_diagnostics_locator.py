"""Regression: DiagnosticRecorder locates decoder layers via the shared
``_layer_locator.get_decoder_layers``, including the legacy GPT-2
``transformer.h`` path that the previous private list missed.

Closes the locator follow-up noted on PR #23.

Author: KeikaJames, 2026-05-06.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from deltamemory.diagnostics import DiagnosticRecorder


@pytest.fixture(scope="module")
def tiny_qwen3():
    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def tiny_gpt2():
    cfg = GPT2Config(
        vocab_size=256,
        n_embd=64,
        n_layer=3,
        n_head=4,
        n_positions=128,
        n_inner=128,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).eval()


def _layer_count(rec: DiagnosticRecorder) -> int:
    return len(rec._find_decoder_layers())


def test_locator_resolves_modern_qwen3(tiny_qwen3):
    rec = DiagnosticRecorder(tiny_qwen3, patcher=None)
    assert _layer_count(rec) == 3


def test_locator_resolves_legacy_gpt2_transformer_h(tiny_gpt2):
    """Previously the diagnostics private list lacked ``transformer.h``;
    after delegating to the shared locator GPT-2 must be supported."""
    rec = DiagnosticRecorder(tiny_gpt2, patcher=None)
    assert _layer_count(rec) == 3


def test_locator_raises_on_bare_module():
    bare = nn.Linear(8, 8)
    rec = DiagnosticRecorder(bare, patcher=None)
    with pytest.raises(RuntimeError, match="could not locate decoder layers"):
        rec._find_decoder_layers()


def test_diagnostics_runs_on_gpt2_residual_hooks(tiny_gpt2):
    """End-to-end: residual_norm signals emit on a forward through GPT-2."""
    rec = DiagnosticRecorder(tiny_gpt2, patcher=None)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    with rec:
        with torch.no_grad():
            tiny_gpt2(input_ids)
    df = rec.to_pandas()
    residual = df[df["signal_name"] == "residual_norm"]
    assert len(residual) > 0, "no residual_norm signals emitted on GPT-2"
    assert set(residual["layer"].unique()) == {0, 1, 2}
