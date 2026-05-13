"""CPU smoke tests for deltamemory.diagnostics.extra_probes."""

import math

import torch

from deltamemory.probes.extra_probes import (
    bank_eldest_score_margin,
    top_k_softmax_entropy,
)


def test_top_k_entropy_uniform_topk():
    # 8 entries, all equal -> top-k of any k re-normalizes to 1/k -> log(k)
    w = torch.full((1, 8), 1 / 8)
    h = top_k_softmax_entropy(w, k=4)
    assert h.shape == (1,)
    assert math.isclose(float(h[0]), math.log(4), rel_tol=1e-5, abs_tol=1e-5)


def test_top_k_entropy_peaked():
    # one-hot -> entropy of top-k is 0 regardless of k
    w = torch.zeros(1, 16)
    w[0, 3] = 1.0
    h = top_k_softmax_entropy(w, k=8)
    assert float(h[0]) < 1e-5


def test_top_k_entropy_handles_k_larger_than_n():
    w = torch.softmax(torch.randn(2, 5), dim=-1)
    h = top_k_softmax_entropy(w, k=100)
    # equivalent to full-distribution entropy when k > n
    full = -(w * w.clamp_min(1e-12).log()).sum(dim=-1)
    assert torch.allclose(h, full, atol=1e-5)


def test_bank_eldest_score_margin_smoke():
    class FakeTokenizer:
        def encode(self, s, add_special_tokens=True, return_tensors=None):
            ids = [ord(c) % 8 for c in s][:4] or [1]
            if return_tensors == "pt":
                return torch.tensor([ids])
            return ids

    def fwd(ids, bank):
        # logits favor token 5 over token 3
        v = 8
        out = torch.zeros(1, ids.shape[1], v)
        out[..., 5] = 2.0
        out[..., 3] = 1.0
        return out

    margin = bank_eldest_score_margin(
        bank=None,
        forward_with_bank=fwd,
        tokenizer=FakeTokenizer(),
        prompt="abcde",
        target_canonical="\x03",   # ord 3 % 8 = 3
        target_eldest="\x05",      # ord 5 % 8 = 5
    )
    assert margin > 0  # eldest favored


def test_bank_eldest_score_margin_empty_target():
    class T:
        def encode(self, s, add_special_tokens=True, return_tensors=None):
            if return_tensors == "pt":
                return torch.zeros(1, 1, dtype=torch.long)
            return []

    out = bank_eldest_score_margin(
        bank=None, forward_with_bank=lambda i, b: torch.zeros(1, 1, 4),
        tokenizer=T(), prompt="x", target_canonical="", target_eldest="",
    )
    assert math.isnan(out)
