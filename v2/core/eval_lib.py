"""Standard evaluation helpers for v2 experiments."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def nll_on_answer(logits: torch.Tensor, input_ids: torch.Tensor, ans_start: int) -> float:
    """Mean per-token causal NLL on the answer span [ans_start, T)."""
    pred = logits[0, ans_start - 1: -1, :]
    gold = input_ids[0, ans_start:]
    return F.cross_entropy(pred.float(), gold).item()


def encode_qa(tok, prompt: str, target: str, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (enc, prompt_ids, ans_start) — full = prompt + ' ' + target."""
    full = prompt + " " + target
    enc = tok(full, return_tensors="pt").to(device)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids
    return enc, prompt_ids, prompt_ids.shape[1]


def mean_nll(model, tok, items: Iterable[tuple[str, str, str]], device) -> float:
    """Compute mean answer-NLL of bare base model over (subj, rel, target) items."""
    nlls = []
    for sj, rl, tg in items:
        prompt = f"{sj} {rl}"
        enc, _, ans_start = encode_qa(tok, prompt, tg, device)
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits
        nlls.append(nll_on_answer(logits, enc.input_ids, ans_start))
    return sum(nlls) / max(len(nlls), 1)
