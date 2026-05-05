"""Cross-architecture generality tests for the LOPI residual profiler.

The profiler operates on the residual-stream norm sequence
``hidden_states[1:]`` returned by HF causal LMs.  This test file pins the
contract that the profiler is *architecture-agnostic* with respect to:

* GQA / MQA — head structure is invisible at the residual stream
* MoE — sparse routing leaves residual shape (B, T, D) unchanged
* Post-attn-norm vs pre-attn-norm convention — block-output norms still
  reflect the residual scale at the layer boundary
* Layer-count scaling — μ_arch is a dimensionless layer index that must
  be in [0, L) for any L

These are pinning regressions: if a future refactor accidentally adds a
layer-count-aware constant (e.g. ``mu_arch = 0.7 * L``) or a
head-dim-aware shape assumption, the corresponding test fails.

The fixtures here are purposely shallow — full-stack HF models are not
needed because ``profile_residuals`` only consumes the residual norms
that any HF causal LM exposes via ``output_hidden_states=True``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from deltamemory.memory.lopi_profiler import (
    LOPIProfile,
    profile_residuals,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal HF-look-alikes that vary along one architectural axis
# ---------------------------------------------------------------------------


class _Tok:
    """Tiny tokenizer compatible with the profiler call signature."""

    pad_token_id = 0
    eos_token_id = 0
    pad_token = "[PAD]"
    eos_token = "[PAD]"

    def __call__(self, prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=32):
        ids = [[min(ord(c) % 50 + 1, 50) for c in s][:max_length] for s in prompts]
        L = max((len(t) for t in ids), default=1)
        padded = [t + [0] * (L - len(t)) for t in ids]
        mask = [[1] * len(t) + [0] * (L - len(t)) for t in ids]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class _Out:
    def __init__(self, hidden_states):
        self.hidden_states = tuple(hidden_states)
        self.logits = None


class _DenseGQABlock(nn.Module):
    """Block whose internal attn uses GQA (n_kv_heads != n_q_heads).

    For the profiler the only thing that matters is the *residual output
    shape*, not the head structure.  We model that by returning ``x +
    f(x)`` where ``f`` is a dense linear that does NOT depend on head
    layout.
    """

    def __init__(self, hidden: int, *, scale: float):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden) * scale)

    def forward(self, x):
        return x + self.linear(x)


class _MoEBlock(nn.Module):
    """Sparse-MoE block that mimics Mixtral / Qwen3-MoE residual semantics.

    Routes each token to one of ``num_experts`` experts via a soft
    ``argmax`` (``topk=1``) and produces the residual update.  The
    profiler only sees the residual sum, so this is a tight contract.
    """

    def __init__(self, hidden: int, *, num_experts: int = 4, scale: float):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(num_experts)]
        )
        with torch.no_grad():
            for i, e in enumerate(self.experts):
                e.weight.copy_(torch.eye(hidden) * (scale * (1.0 + 0.05 * i)))

    def forward(self, x):
        # (B, T, D) -> (B, T, E) routing scores
        scores = self.gate(x)
        # top-1 expert per token
        chosen = scores.argmax(dim=-1)  # (B, T)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (chosen == i).unsqueeze(-1).to(x.dtype)
            out = out + mask * expert(x)
        return x + out


class _PostNormBlock(nn.Module):
    """Block applying RMSNorm AFTER the residual add (Gemma3 / GPT-2 style).

    The residual *output* still represents the layer-boundary residual
    state; the profiler measures its norm uniformly.
    """

    def __init__(self, hidden: int, *, scale: float):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden) * scale)

    def forward(self, x):
        return self.norm(x + self.linear(x))


class _ConfigurableModel(nn.Module):
    """HF-shaped causal LM whose blocks are user-supplied.

    ``name_or_path`` is set so ``LOPIProfile.model_name`` is populated.
    """

    def __init__(self, blocks: list[nn.Module], hidden: int, vocab: int = 64,
                 *, name: str = "_fake/cross-arch"):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(blocks)
        self.name_or_path = name

    def forward(self, *, input_ids, attention_mask=None,
                output_hidden_states=False, use_cache=False, return_dict=True):
        x = self.embed(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        states = [x]
        for blk in self.blocks:
            x = blk(x)
            states.append(x)
        return _Out(states)


# ---------------------------------------------------------------------------
# Generality tests
# ---------------------------------------------------------------------------


def _peaked_scales(num_layers: int, peak: int) -> list[float]:
    """Per-layer scale where layer ``peak`` has the largest variance."""
    return [1.0 + 0.4 * (1.0 - abs(i - peak) / max(num_layers, 1)) for i in range(num_layers)]


def test_profiler_handles_dense_gqa_shape():
    hidden = 32
    L = 6
    scales = _peaked_scales(L, peak=3)
    blocks = [_DenseGQABlock(hidden, scale=s) for s in scales]
    model = _ConfigurableModel(blocks, hidden, name="_fake/dense-gqa-6L")

    profile = profile_residuals(model, _Tok(), device="cpu")
    assert isinstance(profile, LOPIProfile)
    assert profile.num_layers == L
    assert len(profile.mu_base) == L and len(profile.sigma_base) == L
    assert 0 <= profile.mu_arch < L


def test_profiler_handles_moe_routing():
    hidden = 32
    L = 5
    scales = _peaked_scales(L, peak=2)
    blocks = [_MoEBlock(hidden, num_experts=4, scale=s) for s in scales]
    model = _ConfigurableModel(blocks, hidden, name="_fake/moe-5L-4E")

    profile = profile_residuals(model, _Tok(), device="cpu")
    assert profile.num_layers == L
    # MoE residuals must be finite — no NaNs from sparse routing
    assert all(s == s and s >= 0.0 for s in profile.sigma_base)
    assert all(m == m and m >= 0.0 for m in profile.mu_base)
    assert 0 <= profile.mu_arch < L


def test_profiler_handles_post_norm_convention():
    hidden = 32
    L = 4
    blocks = [_PostNormBlock(hidden, scale=1.0 + 0.1 * i) for i in range(L)]
    model = _ConfigurableModel(blocks, hidden, name="_fake/post-norm-4L")

    profile = profile_residuals(model, _Tok(), device="cpu")
    assert profile.num_layers == L
    # Post-norm caps the residual scale; sigma should be small but well-defined
    assert all(s >= 0.0 for s in profile.sigma_base)
    assert 0 <= profile.mu_arch < L


def test_mu_arch_index_independent_of_layer_count():
    """mu_arch is an index in [0, L); doubling L must not push it out of range."""
    hidden = 32
    for L in (2, 8, 24, 80):
        scales = _peaked_scales(L, peak=L // 3)
        blocks = [_DenseGQABlock(hidden, scale=s) for s in scales]
        model = _ConfigurableModel(blocks, hidden, name=f"_fake/scaled-{L}L")
        profile = profile_residuals(model, _Tok(), device="cpu")
        assert profile.num_layers == L
        assert 0 <= profile.mu_arch < L, f"mu_arch out of range for L={L}"
        assert len(profile.mu_base) == L
        assert len(profile.sigma_base) == L


def test_profiler_runs_in_bfloat16_and_returns_float_stats():
    """bf16 model weights must not poison the float64 stats."""
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        # bf16 on CPU is functional in modern PyTorch; just exercise the path
        pass
    hidden = 32
    L = 4
    blocks = [_DenseGQABlock(hidden, scale=1.0 + 0.1 * i) for i in range(L)]
    model = _ConfigurableModel(blocks, hidden, name="_fake/bf16-4L").to(torch.bfloat16)

    profile = profile_residuals(model, _Tok(), device="cpu")
    assert profile.num_layers == L
    # ``dtype`` field reflects model parameter dtype
    assert profile.dtype == "bfloat16"
    # Stats are plain Python floats (json-serialisable), regardless of model dtype
    assert all(isinstance(x, float) for x in profile.mu_base)
    assert all(isinstance(x, float) for x in profile.sigma_base)


def test_profiler_weights_bit_equal_across_archs():
    """Profile is forward-only on every architectural variant."""
    hidden = 32
    cases = [
        ("dense", [_DenseGQABlock(hidden, scale=1.0 + 0.1 * i) for i in range(4)]),
        ("moe", [_MoEBlock(hidden, num_experts=3, scale=1.0 + 0.1 * i) for i in range(4)]),
        ("post_norm", [_PostNormBlock(hidden, scale=1.0 + 0.1 * i) for i in range(4)]),
    ]
    for name, blocks in cases:
        model = _ConfigurableModel(blocks, hidden, name=f"_fake/{name}-bit-equal")
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        profile_residuals(model, _Tok(), device="cpu")
        after = model.state_dict()
        for k, v_before in before.items():
            assert torch.equal(v_before, after[k]), (
                f"weight mutated by profile_residuals on arch={name}, key={k}"
            )
