"""Tests for HTWR injector (Exp12)."""
from __future__ import annotations

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from deltamemory.memory.htwr_injector import (
    HTWRConfig,
    HTWRInjector,
    HTWRMemoryBank,
    OracleRetriever,
    RandomRetriever,
    RawCosineRetriever,
    WritePrompt,
    build_bank,
)

_VOCAB = 128
_LAYERS = 3
_HIDDEN = 32
_HEADS = 4


@pytest.fixture()
def tiny_gpt2():
    cfg = GPT2Config(
        vocab_size=_VOCAB, n_embd=_HIDDEN, n_layer=_LAYERS, n_head=_HEADS,
        n_positions=64, n_inner=64,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    return GPT2LMHeadModel(cfg).eval()


def _ids(seed: int = 0, length: int = 12) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, _VOCAB, (1, length), generator=gen)


# ---------------------------------------------------------------------------
# Capture / inject basics
# ---------------------------------------------------------------------------


def test_capture_shape(tiny_gpt2):
    inj = HTWRInjector(tiny_gpt2, HTWRConfig(hook_point="block_output"))
    mem = inj.capture(_ids())
    assert mem.shape == (_LAYERS, _HIDDEN)
    assert mem.dtype == torch.float32


def test_eta_zero_bit_equal(tiny_gpt2):
    inj = HTWRInjector(tiny_gpt2, HTWRConfig(eta=0.0, hook_point="block_output"))
    mem = inj.capture(_ids())
    ids = _ids(1)
    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
    out, diag = inj.forward_with_memory(mem, input_ids=ids)
    assert torch.equal(base, out.logits)
    assert diag["htwr_eta_effective"] == 0.0


def test_injection_changes_output(tiny_gpt2):
    inj = HTWRInjector(tiny_gpt2, HTWRConfig(eta=0.5, hook_point="block_output"))
    mem = inj.capture(_ids(5))
    ids = _ids(6)
    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
    out, _ = inj.forward_with_memory(mem, input_ids=ids)
    assert not torch.equal(base, out.logits)


def test_sign_flip(tiny_gpt2):
    inj_pos = HTWRInjector(tiny_gpt2, HTWRConfig(eta=0.5, inject_sign=1.0))
    inj_neg = HTWRInjector(tiny_gpt2, HTWRConfig(eta=0.5, inject_sign=-1.0))
    mem = inj_pos.capture(_ids(7))
    ids = _ids(8)
    out_p, _ = inj_pos.forward_with_memory(mem, input_ids=ids)
    out_n, _ = inj_neg.forward_with_memory(mem, input_ids=ids)
    assert not torch.equal(out_p.logits, out_n.logits)


def test_inject_layers_subset(tiny_gpt2):
    """inject_layers=() should be bit-equal even with nonzero eta."""
    inj_none = HTWRInjector(
        tiny_gpt2, HTWRConfig(eta=0.5, inject_layers=())
    )
    mem = inj_none.capture(_ids(9))
    ids = _ids(10)
    with torch.no_grad():
        base = tiny_gpt2(input_ids=ids, use_cache=False).logits.detach().clone()
    out, _ = inj_none.forward_with_memory(mem, input_ids=ids)
    assert torch.equal(base, out.logits)


def test_hooks_detach(tiny_gpt2):
    inj = HTWRInjector(tiny_gpt2, HTWRConfig(eta=0.2))
    mem = inj.capture(_ids(11))
    before_pre = [len(l._forward_pre_hooks) for l in inj.layers]
    before_post = [len(l._forward_hooks) for l in inj.layers]
    inj.forward_with_memory(mem, input_ids=_ids(12))
    after_pre = [len(l._forward_pre_hooks) for l in inj.layers]
    after_post = [len(l._forward_hooks) for l in inj.layers]
    assert before_pre == after_pre
    assert before_post == after_post


# ---------------------------------------------------------------------------
# Bank + retrievers
# ---------------------------------------------------------------------------


def _make_bank(inj: HTWRInjector, n: int = 4) -> HTWRMemoryBank:
    prompts = [
        WritePrompt(fact_id=f"f{i}", input_ids=_ids(100 + i, length=10))
        for i in range(n)
    ]
    return build_bank(inj, prompts)


def test_build_bank_shapes(tiny_gpt2):
    inj = HTWRInjector(tiny_gpt2)
    bank = _make_bank(inj, n=5)
    assert bank.memories.shape == (5, _LAYERS, _HIDDEN)
    assert bank.fact_ids == [f"f{i}" for i in range(5)]
    assert bank.n_memories == 5


def test_oracle_retriever(tiny_gpt2):
    inj = HTWRInjector(tiny_gpt2)
    bank = _make_bank(inj, n=4)
    q = inj.query_residuals(_ids(50))
    r = OracleRetriever().retrieve(q, bank, correct_fact_id="f2")
    assert r.top_index == 2
    assert r.retrieval_accuracy is True


def test_raw_cosine_retriever_accuracy_on_orthogonal_keys():
    """Synthetic bank: each fact's key is a one-hot direction at every layer.
    Query == one of those one-hots should retrieve that fact perfectly."""
    n_layers, hidden, n_facts = 3, 16, 4
    bank_keys = torch.zeros(n_facts, n_layers, hidden)
    for i in range(n_facts):
        bank_keys[i, :, i] = 1.0
    bank = HTWRMemoryBank(memories=bank_keys, fact_ids=[f"f{i}" for i in range(n_facts)])
    retr = RawCosineRetriever()
    # Query "f2"
    q = torch.zeros(n_layers, hidden)
    q[:, 2] = 1.0
    r = retr.retrieve(q, bank, correct_fact_id="f2")
    assert r.top_index == 2
    assert r.retrieval_accuracy is True
    assert r.top_score == pytest.approx(1.0)


def test_random_retriever_seeded():
    n_layers, hidden, n_facts = 2, 8, 5
    bank = HTWRMemoryBank(
        memories=torch.randn(n_facts, n_layers, hidden),
        fact_ids=[f"f{i}" for i in range(n_facts)],
    )
    r1 = RandomRetriever(seed=42).retrieve(torch.zeros(n_layers, hidden), bank)
    r2 = RandomRetriever(seed=42).retrieve(torch.zeros(n_layers, hidden), bank)
    assert r1.top_index == r2.top_index


def test_bank_shuffled_layers_preserves_norms():
    n_layers, hidden, n_facts = 4, 8, 3
    mem = torch.randn(n_facts, n_layers, hidden)
    bank = HTWRMemoryBank(memories=mem, fact_ids=[f"f{i}" for i in range(n_facts)])
    sh = bank.shuffled_layers(seed=7)
    for i in range(n_facts):
        orig_norms = sorted(mem[i].norm(dim=-1).tolist())
        sh_norms = sorted(sh.memories[i].norm(dim=-1).tolist())
        assert orig_norms == pytest.approx(sh_norms, rel=1e-6)


def test_end_to_end_oracle_inject_changes_output(tiny_gpt2):
    """Pipeline smoke: build bank → query → oracle retrieve → inject."""
    inj = HTWRInjector(tiny_gpt2, HTWRConfig(eta=0.3))
    bank = _make_bank(inj, n=3)
    q = inj.query_residuals(_ids(77))
    r = OracleRetriever().retrieve(q, bank, correct_fact_id="f1")
    mem = bank.index(r.top_index)
    with torch.no_grad():
        base = tiny_gpt2(input_ids=_ids(77), use_cache=False).logits.detach().clone()
    out, _ = inj.forward_with_memory(mem, input_ids=_ids(77))
    assert not torch.equal(base, out.logits)
