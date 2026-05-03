"""Tests for Stage 14B/14C address-conditional capture policy."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from deltamemory.memory.capture_policy import (
    CaptureSite,
    resolve_capture_sites,
)


class _ToyTokenizer:
    """Word-level tokenizer that returns ids = [1] + [10+i for each word].

    Special-token offset is 1 so positions for the regex-matched address
    tokens are deterministic and easy to verify.
    """

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        words = text.split()
        ids = [1] + [10 + i for i in range(len(words))] if add_special_tokens else [
            10 + i for i in range(len(words))
        ]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.tensor([[1] * len(ids)]),
            }
        if return_tensors is None:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        raise NotImplementedError(return_tensors)


def _mask_for(n: int) -> torch.Tensor:
    return torch.tensor([1] * n)


def test_period_policy_returns_last_token_only() -> None:
    tok = _ToyTokenizer()
    prompt = "The capital of France is Paris ."
    n_tokens = len(tok(prompt, return_tensors=None)["input_ids"])
    sites = resolve_capture_sites(
        policy="period",
        write_prompt=prompt,
        address="The capital of France",
        tokenizer=tok,
        attention_mask_row=_mask_for(n_tokens),
    )
    assert sites == [CaptureSite(n_tokens - 1, "period")]


def test_address_policy_uses_address_span_position() -> None:
    tok = _ToyTokenizer()
    prompt = "The capital of France is Paris ."
    address = "The capital of France"
    addr_ids = tok(address, return_tensors=None, add_special_tokens=True)["input_ids"]
    expected_addr_pos = len(addr_ids) - 1

    n_tokens = len(tok(prompt, return_tensors=None)["input_ids"])
    sites = resolve_capture_sites(
        policy="address",
        write_prompt=prompt,
        address=address,
        tokenizer=tok,
        attention_mask_row=_mask_for(n_tokens),
    )
    assert sites == [CaptureSite(expected_addr_pos, "address")]
    assert sites[0].token_pos < n_tokens - 1


def test_multi_policy_returns_address_then_period() -> None:
    tok = _ToyTokenizer()
    prompt = "The capital of France is Paris ."
    address = "The capital of France"
    n_tokens = len(tok(prompt, return_tensors=None)["input_ids"])
    sites = resolve_capture_sites(
        policy="multi",
        write_prompt=prompt,
        address=address,
        tokenizer=tok,
        attention_mask_row=_mask_for(n_tokens),
    )
    assert len(sites) == 2
    assert sites[0].role == "address"
    assert sites[1].role == "period"
    assert sites[0].token_pos < sites[1].token_pos


def test_address_policy_falls_back_to_period_when_unresolvable() -> None:
    tok = _ToyTokenizer()
    prompt = "Some sentence without obvious address structure."
    n_tokens = len(tok(prompt, return_tensors=None)["input_ids"])
    sites = resolve_capture_sites(
        policy="address",
        write_prompt=prompt,
        address=None,
        tokenizer=tok,
        attention_mask_row=_mask_for(n_tokens),
    )
    assert sites == [CaptureSite(n_tokens - 1, "period")]


def test_regex_fallback_when_address_missing() -> None:
    tok = _ToyTokenizer()
    prompt = "The capital of France is Paris."
    n_tokens = len(tok(prompt, return_tensors=None)["input_ids"])
    sites = resolve_capture_sites(
        policy="address",
        write_prompt=prompt,
        address=None,
        tokenizer=tok,
        attention_mask_row=_mask_for(n_tokens),
    )
    # Regex extracts "The capital of France" and yields its last-token pos
    # which must be strictly before the period token.
    assert sites[0].role == "address"
    assert sites[0].token_pos < n_tokens - 1


def test_unknown_policy_raises() -> None:
    tok = _ToyTokenizer()
    try:
        resolve_capture_sites(
            policy="bogus",
            write_prompt="x is y.",
            address="x",
            tokenizer=tok,
            attention_mask_row=_mask_for(3),
        )
    except ValueError as exc:
        assert "bogus" in str(exc)
        return
    raise AssertionError("expected ValueError for unknown policy")
