"""HF-transformers reference adapter for MnemeLogitsProcessor.

Runs the same ``MnemeLogitsProcessor`` against an HF model in a manual
greedy-decode loop so we can produce a bit-equality witness vs vLLM later.

Usage::

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mneme_vllm import MnemeLogitsProcessor
    from mneme_vllm.hf_reference import hf_run_with_processor

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tok   = AutoTokenizer.from_pretrained("gpt2")
    mp    = MnemeLogitsProcessor(alpha=0.0)

    text = hf_run_with_processor(model, tok, "Hello world", mp, max_new_tokens=8)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass  # avoid hard HF import at type-check time


def hf_run_with_processor(
    model: "torch.nn.Module",
    tok: object,
    prompt: str,
    processor: object,
    *,
    max_new_tokens: int = 16,
) -> str:
    """Greedy-decode ``prompt`` using ``model``, applying ``processor`` at each step.

    Parameters
    ----------
    model:
        Any HF ``PreTrainedModel`` (or duck-typed equivalent) with a
        ``forward(**inputs) -> CausalLMOutput`` interface.
    tok:
        HF tokenizer with ``encode`` / ``decode`` methods.
    prompt:
        Plain-text prompt string.
    processor:
        Any callable with signature
        ``(token_ids: list[int], logits: torch.Tensor) -> torch.Tensor``.
        Typically a ``MnemeLogitsProcessor`` instance.
    max_new_tokens:
        Maximum number of tokens to generate (greedy only).

    Returns
    -------
    str
        The decoded generated text (prompt tokens excluded).
    """
    device = next(model.parameters()).device
    input_ids: torch.Tensor = tok.encode(prompt, return_tensors="pt").to(device)  # type: ignore[attr-defined]

    generated_ids: list[int] = input_ids[0].tolist()

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=torch.tensor([generated_ids], device=device))
            # outputs.logits shape: (1, seq_len, vocab_size)
            next_token_logits: torch.Tensor = outputs.logits[0, -1, :]

            # Apply processor (vLLM-compatible signature)
            next_token_logits = processor(generated_ids, next_token_logits)  # type: ignore[operator]

            next_token_id: int = int(torch.argmax(next_token_logits).item())
            generated_ids.append(next_token_id)

            # Stop on EOS if the tokenizer has one
            eos_id = getattr(tok, "eos_token_id", None)
            if eos_id is not None and next_token_id == eos_id:
                break

    prompt_len = input_ids.shape[1]
    new_ids = generated_ids[prompt_len:]
    return tok.decode(new_ids, skip_special_tokens=True)  # type: ignore[attr-defined]
