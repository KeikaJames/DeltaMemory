"""Regression tests for Exp11 RSM scoring semantics."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from deltamemory.memory.rsm_injector import RSMConfig, RSMInjector, RSMMemoryBank
from experiments.atb_validation_v1.exp11_rsm_residual_stream_memory.run import (
    _continuation_logp_rsm,
    _phase_b_verdict,
    _run_one_config,
)


class _TinyTokenizer:
    def __init__(self) -> None:
        self.vocab = {"<bos>": 0}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = []
        if add_special_tokens:
            ids.append(0)
        for tok in text.strip().split():
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
            ids.append(self.vocab[tok])
        return ids


class _SpyRSM(RSMInjector):
    def __init__(self, model) -> None:
        super().__init__(model, RSMConfig(eta=0.0, theta=-1.0))
        self.score_lengths: list[int] = []
        self.forward_lengths: list[int] = []

    def score(self, bank, input_ids, attention_mask=None):
        self.score_lengths.append(int(input_ids.shape[1]))
        return torch.ones(bank.n_memories, dtype=torch.float32)

    def forward_with_scores(self, bank, scores, input_ids, attention_mask=None, **kwargs):
        self.forward_lengths.append(int(input_ids.shape[1]))
        return self.model(input_ids=input_ids, use_cache=False), {
            "rsm_activation_rate": 1.0,
            "rsm_max_score": 1.0,
            "rsm_top_fact_id": bank.fact_ids[0],
        }


def test_continuation_rsm_scores_prompt_only_and_injects_prefixes():
    cfg = GPT2Config(
        vocab_size=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_positions=32,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(cfg).eval()
    tok = _TinyTokenizer()
    rsm = _SpyRSM(model)
    bank = RSMMemoryBank(torch.ones(1, 2, 16), ["fact_0"])

    prompt = "subject relation"
    target = "target token"
    prompt_len = len(tok.encode(prompt, add_special_tokens=True))
    full_len = len(tok.encode(f"{prompt} {target}", add_special_tokens=True))

    logp, target_ids, diag = _continuation_logp_rsm(
        rsm, bank, tok, prompt, target, device="cpu"
    )

    assert logp == logp
    assert len(target_ids) == 2
    assert diag["rsm_top_fact_id"] == "fact_0"
    assert rsm.score_lengths == [prompt_len]
    assert rsm.forward_lengths == [prompt_len, prompt_len + 1]
    assert full_len not in rsm.score_lengths


def test_phase_b_verdict_requires_beating_random_memory():
    assert _phase_b_verdict({
        "correct_memory": 1.0,
        "base_model": 0.0,
        "random_memory": 1.2,
        "gap": 0.1,
    }) == "FAIL"
    assert _phase_b_verdict({
        "correct_memory": 1.0,
        "base_model": 0.0,
        "random_memory": 0.9,
        "gate_off": 1.2,
        "gap": -0.2,
    }) == "STABILIZER_ONLY"


def test_include_anb_best_is_disabled_until_full_a3_plumbing():
    with pytest.raises(RuntimeError, match="include-anb-best is disabled"):
        _run_one_config(
            model=None,
            tok=None,
            rsm=None,
            rows=[],
            cache={},
            device="cpu",
            out_dir=Path("/tmp/unused-rsm-test"),
            eta=0.1,
            theta=0.5,
            bank_size=1,
            seeds=[0],
            include_anb_best=True,
        )
