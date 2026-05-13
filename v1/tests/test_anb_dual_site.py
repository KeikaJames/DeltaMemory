"""Tests for anb_dual_site.write_fact_dual_site."""
from __future__ import annotations

import pytest
import torch

from deltamemory.memory.attn_native_bank import (
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.anb_dual_site import write_fact_dual_site


pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


@pytest.fixture(scope="module")
def patcher_tok():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("mps")
    p = AttnNativePatcher(m); p.install()
    yield p, tok
    p.remove(); del m


def _resolve_pos(tok, prompt, role: str) -> int:
    """Pick a stable mid-prompt token position for testing."""
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    am = enc["attention_mask"][0]
    n = int(am.sum().item())
    return {"early": 1, "mid": n // 2, "late": n - 2}[role]


def test_dual_site_equal_pos_matches_write_fact(patcher_tok):
    """pos_K == pos_V must produce a slot bit-equal to write_fact."""
    patcher, tok = patcher_tok
    prompt = "The mother tongue of Marie Curie is Polish."
    pos = _resolve_pos(tok, prompt, "mid")

    bank_a = fresh_bank(patcher.model)
    bank_a.value_scale_mode = "auto_rms_cap"; bank_a.bank_key_mode = "pre_rope"
    write_fact(patcher, bank_a, tok, write_prompt=prompt, fact_id="A",
               address=None, capture_pos=pos)

    bank_b = fresh_bank(patcher.model)
    bank_b.value_scale_mode = "auto_rms_cap"; bank_b.bank_key_mode = "pre_rope"
    write_fact_dual_site(patcher, bank_b, tok, write_prompt=prompt, fact_id="A",
                         address=None, capture_pos_K=pos, capture_pos_V=pos)

    assert len(bank_a.fact_ids) == len(bank_b.fact_ids) == 1
    for layer in range(bank_a.num_layers):
        assert torch.equal(bank_a.M_K[layer], bank_b.M_K[layer]), f"K diff at layer {layer}"
        assert torch.equal(bank_a.M_V[layer], bank_b.M_V[layer]), f"V diff at layer {layer}"


def test_dual_site_distinct_pos_one_slot_correct_sources(patcher_tok):
    """pos_K != pos_V: K matches single-pos@pos_K, V matches single-pos@pos_V."""
    patcher, tok = patcher_tok
    prompt = "The mother tongue of Marie Curie is Polish."
    pos_K = _resolve_pos(tok, prompt, "early")
    pos_V = _resolve_pos(tok, prompt, "late")
    assert pos_K != pos_V

    ref_K = fresh_bank(patcher.model)
    ref_K.value_scale_mode = "auto_rms_cap"; ref_K.bank_key_mode = "pre_rope"
    write_fact(patcher, ref_K, tok, write_prompt=prompt, fact_id="A",
               address=None, capture_pos=pos_K)

    ref_V = fresh_bank(patcher.model)
    ref_V.value_scale_mode = "auto_rms_cap"; ref_V.bank_key_mode = "pre_rope"
    write_fact(patcher, ref_V, tok, write_prompt=prompt, fact_id="A",
               address=None, capture_pos=pos_V)

    bank = fresh_bank(patcher.model)
    bank.value_scale_mode = "auto_rms_cap"; bank.bank_key_mode = "pre_rope"
    write_fact_dual_site(patcher, bank, tok, write_prompt=prompt, fact_id="A",
                         address=None, capture_pos_K=pos_K, capture_pos_V=pos_V)

    assert len(bank.fact_ids) == 1  # one slot, not two
    for layer in range(bank.num_layers):
        assert torch.equal(bank.M_K[layer], ref_K.M_K[layer]), f"K mismatch L{layer}"
        assert torch.equal(bank.M_V[layer], ref_V.M_V[layer]), f"V mismatch L{layer}"


def test_dual_site_slot_count_n_facts(patcher_tok):
    """Writing N facts must produce N slots (not 2N)."""
    patcher, tok = patcher_tok
    prompt = "The mother tongue of Marie Curie is Polish."
    pK = _resolve_pos(tok, prompt, "early"); pV = _resolve_pos(tok, prompt, "late")
    bank = fresh_bank(patcher.model)
    bank.value_scale_mode = "auto_rms_cap"; bank.bank_key_mode = "pre_rope"
    for i in range(3):
        write_fact_dual_site(patcher, bank, tok, write_prompt=prompt,
                             fact_id=f"F{i}", address=None,
                             capture_pos_K=pK, capture_pos_V=pV)
    assert len(bank.fact_ids) == 3
    for layer in range(bank.num_layers):
        assert bank.M_K[layer].shape[0] == 3
        assert bank.M_V[layer].shape[0] == 3
