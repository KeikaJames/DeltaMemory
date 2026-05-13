"""Unit tests for Exp17 extended capture-site resolver.

Uses the GPT-2 tokenizer (small, cached) so prefix-length resolution is
exercised on a real tokenizer that does NOT use BOS by default; we still
exercise both add_special_tokens settings.
"""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer  # noqa: E402

from deltamemory.memory.anb_capture_sweep import (
    CaptureSpec,
    VALID_SITES,
    derive_relation_phrase,
    resolve_extended_capture,
)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("gpt2")


def _attn_mask(tok, prompt: str, add_special: bool):
    out = tok(prompt, return_tensors="pt", add_special_tokens=add_special)
    return out["attention_mask"][0]


def test_derive_relation_phrase():
    assert derive_relation_phrase("{} is located in") == "is located in"
    assert derive_relation_phrase("Madrid is in {}") is not None
    assert derive_relation_phrase(None) is None
    assert derive_relation_phrase("") is None


def test_valid_sites_listed():
    assert "period" in VALID_SITES
    assert "subject_last" in VALID_SITES
    assert "all_content_sparse" in VALID_SITES


def test_period_is_last_real_token(tok):
    prompt = "The Eiffel Tower is located in Paris."
    mask = _attn_mask(tok, prompt, add_special=False)
    spec = resolve_extended_capture(
        site="period",
        write_prompt=prompt,
        subject="The Eiffel Tower",
        relation_phrase="is located in",
        object_str="Paris",
        tokenizer=tok,
        attention_mask_row=mask,
        add_special_tokens=False,
    )
    assert spec is not None
    expected = int(mask.sum().item()) - 1
    assert spec.token_positions == [expected]


def test_subject_first_and_last(tok):
    prompt = "The Eiffel Tower is located in Paris."
    mask = _attn_mask(tok, prompt, add_special=False)
    first = resolve_extended_capture(
        site="subject_first", write_prompt=prompt,
        subject="The Eiffel Tower", relation_phrase="is located in",
        object_str="Paris", tokenizer=tok, attention_mask_row=mask,
        add_special_tokens=False,
    )
    last = resolve_extended_capture(
        site="subject_last", write_prompt=prompt,
        subject="The Eiffel Tower", relation_phrase="is located in",
        object_str="Paris", tokenizer=tok, attention_mask_row=mask,
        add_special_tokens=False,
    )
    assert first is not None and last is not None
    assert first.token_positions[0] < last.token_positions[0]
    # subject occupies tokens 0..N -> first should be 0 for gpt2 (no BOS).
    assert first.token_positions[0] == 0


def test_object_last_is_inside_real_tokens(tok):
    prompt = "The Eiffel Tower is located in Paris."
    mask = _attn_mask(tok, prompt, add_special=False)
    spec = resolve_extended_capture(
        site="object_last", write_prompt=prompt,
        subject="The Eiffel Tower", relation_phrase="is located in",
        object_str="Paris", tokenizer=tok, attention_mask_row=mask,
        add_special_tokens=False,
    )
    assert spec is not None
    period = int(mask.sum().item()) - 1
    assert 0 <= spec.token_positions[0] <= period


def test_pair_returns_two_positions(tok):
    prompt = "The Eiffel Tower is located in Paris."
    mask = _attn_mask(tok, prompt, add_special=False)
    spec = resolve_extended_capture(
        site="subject_relation_pair", write_prompt=prompt,
        subject="The Eiffel Tower", relation_phrase="is located in",
        object_str="Paris", tokenizer=tok, attention_mask_row=mask,
        add_special_tokens=False,
    )
    assert spec is not None
    assert len(spec.token_positions) == 2
    assert spec.token_positions[0] != spec.token_positions[1]


def test_full_content_returns_up_to_three(tok):
    prompt = "The Eiffel Tower is located in Paris."
    mask = _attn_mask(tok, prompt, add_special=False)
    spec = resolve_extended_capture(
        site="full_content", write_prompt=prompt,
        subject="The Eiffel Tower", relation_phrase="is located in",
        object_str="Paris", tokenizer=tok, attention_mask_row=mask,
        add_special_tokens=False,
    )
    assert spec is not None
    assert 1 <= len(spec.token_positions) <= 3


def test_all_content_sparse_dedupes_and_sorts(tok):
    prompt = "The Eiffel Tower is located in Paris."
    mask = _attn_mask(tok, prompt, add_special=False)
    spec = resolve_extended_capture(
        site="all_content_sparse", write_prompt=prompt,
        subject="The Eiffel Tower", relation_phrase="is located in",
        object_str="Paris", tokenizer=tok, attention_mask_row=mask,
        add_special_tokens=False,
    )
    assert spec is not None
    assert spec.token_positions == sorted(set(spec.token_positions))


def test_missing_span_returns_none(tok):
    prompt = "The Eiffel Tower is located in Paris."
    mask = _attn_mask(tok, prompt, add_special=False)
    spec = resolve_extended_capture(
        site="subject_last", write_prompt=prompt,
        subject="Statue of Liberty", relation_phrase="is located in",
        object_str="Paris", tokenizer=tok, attention_mask_row=mask,
        add_special_tokens=False,
    )
    assert spec is None


def test_unknown_site_raises(tok):
    with pytest.raises(ValueError):
        resolve_extended_capture(
            site="not_a_site", write_prompt="x",
            subject="x", relation_phrase=None, object_str="x",
            tokenizer=tok, attention_mask_row=torch.tensor([1]),
            add_special_tokens=False,
        )
