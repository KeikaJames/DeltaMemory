"""ATB metric fast path parity tests."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from experiments.atb_validation_v1._lib import (
    continuation_logp,
    evaluate_prompt,
    first_token_rank,
)


class _ToyTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<bos>": 0}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [0] if add_special_tokens else []
        for token in text.split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return ids


class _ToyCausalLM:
    """Deterministic causal model whose logits depend only on prefix tokens."""

    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(self, input_ids: torch.Tensor, use_cache: bool = False) -> Any:
        del use_cache
        self.calls += 1
        batch, seq_len = input_ids.shape
        vocab_axis = torch.arange(self.vocab_size, device=input_ids.device).float()
        logits = torch.empty(batch, seq_len, self.vocab_size, device=input_ids.device)
        for pos in range(seq_len):
            prefix_sum = input_ids[:, : pos + 1].float().sum(dim=1, keepdim=True)
            logits[:, pos, :] = torch.sin(prefix_sum * 0.17 + vocab_axis * 0.11)
            logits[:, pos, :] += (vocab_axis.remainder(7) * 0.01)
        return SimpleNamespace(logits=logits)


def _legacy_evaluate_prompt(
    model: Any,
    tok: Any,
    prompt: str,
    target_new: str,
    target_true: str,
    device: str,
) -> dict[str, Any]:
    logp_new, ids_new = continuation_logp(model, tok, prompt, target_new, device)
    logp_true, _ = continuation_logp(model, tok, prompt, target_true, device)
    target_new_first = ids_new[0] if ids_new else -1
    rank, _ = first_token_rank(model, tok, prompt, target_new_first, device)
    return {
        "target_new_logprob": logp_new,
        "target_true_logprob": logp_true,
        "margin": logp_new - logp_true,
        "target_rank": rank,
        "recall_at_1": (rank == 0),
    }


def test_evaluate_prompt_fast_path_matches_legacy_metrics() -> None:
    tok = _ToyTokenizer()
    prompt = "The capital of Avalon is"
    target_new = "Eldoria"
    target_true = "Paris"

    legacy_model = _ToyCausalLM()
    fast_model = _ToyCausalLM()

    expected = _legacy_evaluate_prompt(
        legacy_model, tok, prompt, target_new, target_true, "cpu"
    )
    actual = evaluate_prompt(fast_model, tok, prompt, target_new, target_true, "cpu")

    assert actual == expected
    assert legacy_model.calls == 3
    assert fast_model.calls == 2


def test_evaluate_prompt_can_preserve_legacy_forward_sequence() -> None:
    tok = _ToyTokenizer()
    model = _ToyCausalLM()

    evaluate_prompt(
        model,
        tok,
        "The capital of Avalon is",
        "Eldoria",
        "Paris",
        "cpu",
        preserve_forward_sequence=True,
    )

    assert model.calls == 3
