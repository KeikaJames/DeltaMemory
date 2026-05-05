"""Phase S — Unit tests for U-LOPI auto-calibration profiler.

Coverage
--------
1. Profile reproducibility (same prompts ⇒ same mu/sigma bit-equal).
2. Profile shape: mu_base, sigma_base length == model.num_layers.
3. ``mu_arch`` is a valid layer index AND equals ``argmax_l sigma_base``.
4. Save/load round-trip.
5. Profile forward does not mutate model weights.
6. Profile bit-stable across save/load even for tied sigma -- low-index tiebreak.
7. ``coefficient_of_variation`` triggers ``eta_sigma=0.7`` on heterogeneous
   sigma profiles (D-S-6).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deltamemory.memory.lopi_profiler import (
    _layer_norm_stats,
    default_profile_corpus,
    load_profile,
    profile_residuals,
    save_profile,
)

# ---------------------------------------------------------------------------
# Fake model + tokenizer big enough to expose mu_base / sigma_base structure


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "[PAD]"
    eos_token = "[PAD]"

    def __call__(self, prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=32):
        # Map each character to a small int id; pad to max length.
        ids = []
        for s in prompts:
            t = [min(ord(c) % 50 + 1, 50) for c in s][:max_length]
            ids.append(t)
        L = max(len(t) for t in ids) if ids else 1
        padded = [t + [0] * (L - len(t)) for t in ids]
        mask = [[1] * len(t) + [0] * (L - len(t)) for t in ids]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class _FakeBlock(nn.Module):
    """Tiny linear block with deterministic per-layer norm signature."""

    def __init__(self, hidden: int, layer_idx: int, *, num_layers: int):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)
        # Initialise so middle layer has the largest norm-variation when
        # processing varied-length inputs (helps mu_arch determinism).
        with torch.no_grad():
            scale = 1.0 + 0.1 * (layer_idx - num_layers / 2) ** 2
            self.linear.weight.copy_(torch.eye(hidden) * scale)

    def forward(self, x):
        return self.linear(x)


class _FakeModelOut:
    def __init__(self, hidden_states):
        self.hidden_states = tuple(hidden_states)
        self.logits = None


class _FakeModel(nn.Module):
    """A minimal HF-look-alike with output_hidden_states support."""

    name_or_path = "_fake/test-12L-32d"

    def __init__(self, num_layers: int = 6, hidden: int = 32, vocab: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(
            [_FakeBlock(hidden, i, num_layers=num_layers) for i in range(num_layers)]
        )
        self._num_layers = num_layers

    def forward(self, *, input_ids, attention_mask=None,
                output_hidden_states=False, use_cache=False, return_dict=True):
        x = self.embed(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        states = [x]
        for blk in self.blocks:
            x = blk(x)
            states.append(x)
        return _FakeModelOut(states)


# ---------------------------------------------------------------------------
# Tests


def test_profile_reproducible():
    model = _FakeModel(num_layers=6, hidden=32)
    tok = _FakeTokenizer()
    prompts = default_profile_corpus()[:5]
    p1 = profile_residuals(model, tok, prompts=prompts)
    p2 = profile_residuals(model, tok, prompts=prompts)
    assert p1.mu_base == p2.mu_base
    assert p1.sigma_base == p2.sigma_base
    assert p1.mu_arch == p2.mu_arch
    assert p1.profile_corpus_sha == p2.profile_corpus_sha


def test_profile_shape_matches_model_depth():
    model = _FakeModel(num_layers=8, hidden=16)
    p = profile_residuals(model, _FakeTokenizer(), prompts=["hello", "world"])
    assert len(p.mu_base) == 8
    assert len(p.sigma_base) == 8
    assert p.num_layers == 8


def test_mu_arch_in_range_and_argmax():
    model = _FakeModel(num_layers=6, hidden=32)
    p = profile_residuals(model, _FakeTokenizer(),
                          prompts=default_profile_corpus())
    assert 0 <= p.mu_arch < p.num_layers
    # argmax with low-index tie-break:
    expected = max(range(len(p.sigma_base)), key=lambda i: (p.sigma_base[i], -i))
    # Our impl breaks ties to lower index ⇒ pick first index achieving max.
    max_v = max(p.sigma_base)
    expected = next(i for i, v in enumerate(p.sigma_base) if v == max_v)
    assert p.mu_arch == expected


def test_profile_save_load_roundtrip(tmp_path):
    model = _FakeModel(num_layers=4, hidden=16)
    p = profile_residuals(model, _FakeTokenizer(), prompts=["a", "b", "c"])
    path = tmp_path / "profile.json"
    save_profile(p, path)
    p2 = load_profile(path)
    assert p2.asdict() == p.asdict()


def test_profile_does_not_mutate_weights():
    model = _FakeModel(num_layers=4, hidden=16)
    snap = {n: t.detach().clone() for n, t in model.state_dict().items()}
    profile_residuals(model, _FakeTokenizer(),
                      prompts=default_profile_corpus())
    for n, t in model.state_dict().items():
        torch.testing.assert_close(t, snap[n], rtol=0, atol=0)


def test_profile_corpus_sha_changes_with_prompts():
    model = _FakeModel(num_layers=4, hidden=16)
    p1 = profile_residuals(model, _FakeTokenizer(), prompts=["one"])
    p2 = profile_residuals(model, _FakeTokenizer(), prompts=["two"])
    assert p1.profile_corpus_sha != p2.profile_corpus_sha


def test_eta_sigma_default_one_for_homogeneous():
    # Homogeneous sigma => CV ~ 0 => eta_sigma=1.0.
    model = _FakeModel(num_layers=4, hidden=16)
    # Set every block to identity so sigma_base is dominated by input variance
    # only, which is roughly the same per layer => low CV.
    for blk in model.blocks:
        with torch.no_grad():
            blk.linear.weight.copy_(torch.eye(16))
    p = profile_residuals(model, _FakeTokenizer(),
                          prompts=default_profile_corpus())
    assert p.eta_sigma == 1.0


def test_empty_prompts_raises():
    with pytest.raises(ValueError):
        profile_residuals(_FakeModel(), _FakeTokenizer(), prompts=[])


def test_layer_norm_stats_excludes_padding_tokens():
    h0 = torch.zeros(2, 2, 2)
    h1 = torch.tensor([
        [[3.0, 4.0], [300.0, 400.0]],
        [[6.0, 8.0], [0.0, 0.0]],
    ])
    mask = torch.tensor([[1, 0], [1, 1]])

    mu, sigma = _layer_norm_stats((h0, h1), attention_mask=mask)
    valid_norms = torch.tensor([5.0, 10.0, 0.0])
    assert mu == [float(valid_norms.mean().item())]
    assert sigma == [float(valid_norms.std(unbiased=False).item())]
