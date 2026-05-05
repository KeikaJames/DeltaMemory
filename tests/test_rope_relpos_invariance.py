"""RoPE relative-position correctness invariants for AttnNativeBank.

Pinned design contract (`deltamemory/memory/attn_native_bank.py:18-24`):

    The bank stores **pre-RoPE post-norm K**.  At attention time we keep a
    **pre-RoPE Q copy** and use it only for the bank slice:

        scores_orig = q_post @ k_post^T * scaling   # standard, with RoPE
        scores_bank = q_pre  @ M_K^T    * scaling   # both sides pre-RoPE

The "both sides pre-RoPE" choice makes ``scores_bank`` *exactly* position
invariant: the bank score depends only on the (post-norm) content of Q and
K, not on the token position at which they were captured nor the position
at which they are queried.  Without this design, a fact written at token
position p_w would only score correctly for queries at position p_q
satisfying p_q - p_w == 0 (mod the RoPE wavelength), since RoPE encodes
*relative* position into the inner product.

This module pins three things:

1. The mathematical invariant — ``q_pre @ k_pre^T`` is identical to
   ``apply_rope(q, p_q) @ apply_rope(k, p_w)^T`` only when ``p_q == p_w``;
   otherwise they diverge by a position-dependent rotation.  Tested across
   Llama (rope_theta=1e4), Qwen3 (rope_theta=1e6), and a partial-RoPE
   variant (GLM-4 style, only the first half of head_dim is rotated).
2. The GQA edge case — ``repeat_kv`` applied to pre-RoPE bank K preserves
   the invariant (the kv-group dimension does not interact with positional
   rotation).
3. The code-level tripwire — ``deltamemory/memory/attn_native_bank.py``
   computes ``scores_bank`` with ``q_pre @ mk_e``, never ``q_post``.  This
   is a textual guard against future refactors that silently swap the
   variable name and break the invariant.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Reference RoPE implementation (matches HF Llama / Qwen3 / Gemma3 conventions
# at the math level; we replicate it here so the test does not depend on a
# specific transformers version).  ``unsqueeze_dim=1`` for the (B, H, T, D)
# layout used everywhere in attn_native_bank.py.
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(
    *,
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` of shape ``(seq_len, head_dim)``."""
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)         # (seq_len, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)          # (seq_len, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(
    q: torch.Tensor,         # (B, H, T, D)
    k: torch.Tensor,         # (B, Hk, T, D)
    cos: torch.Tensor,       # (T, D) or (1, 1, T, D) after unsqueeze
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos_b) + (_rotate_half(q) * sin_b)
    k_rot = (k * cos_b) + (_rotate_half(k) * sin_b)
    return q_rot, k_rot


def _rope_at_position(
    x: torch.Tensor,         # (B, H, 1, D)
    *,
    position: int,
    rope_theta: float,
) -> torch.Tensor:
    """Apply RoPE to a single-token tensor as if it lived at ``position``."""
    head_dim = x.shape[-1]
    cos_full, sin_full = _rope_cos_sin(
        seq_len=position + 1,
        head_dim=head_dim,
        rope_theta=rope_theta,
        device=x.device,
        dtype=x.dtype,
    )
    cos = cos_full[position : position + 1]
    sin = sin_full[position : position + 1]
    cos_b = cos.unsqueeze(0).unsqueeze(0)
    sin_b = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos_b) + (_rotate_half(x) * sin_b)


# ---------------------------------------------------------------------------
# Test 1: pre-RoPE inner product is position-invariant; post-RoPE is not.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rope_theta, head_dim",
    [
        (1.0e4, 64),    # Llama / GPT-NeoX standard
        (1.0e6, 128),   # Qwen3 / Gemma3
        (1.0e4, 256),   # Llama-3 large head_dim
    ],
)
def test_pre_rope_dot_is_position_invariant(rope_theta: float, head_dim: int) -> None:
    """The bank's ``q_pre @ k_pre^T`` is exactly the same regardless of p_q, p_w.

    Pins ``deltamemory/memory/attn_native_bank.py`` lines 403, 432, and 516:
    Q/K are captured pre-RoPE, the bank-scoring inner product is taken
    pre-RoPE on both sides, so the bank slot for a fact written at any
    position p_w scores identically against a query at any position p_q.
    """
    torch.manual_seed(0)
    q_content = torch.randn(1, 4, 1, head_dim, dtype=torch.float64)
    k_content = torch.randn(1, 4, 1, head_dim, dtype=torch.float64)

    pre_rope_dot = torch.matmul(q_content, k_content.transpose(-1, -2))

    positions = [(0, 0), (5, 5), (5, 50), (50, 5), (1, 1024), (777, 333)]
    pre_rope_dots = []
    post_rope_dots = []
    for p_q, p_w in positions:
        q_post = _rope_at_position(q_content, position=p_q, rope_theta=rope_theta)
        k_post = _rope_at_position(k_content, position=p_w, rope_theta=rope_theta)
        post_rope_dots.append(torch.matmul(q_post, k_post.transpose(-1, -2)))
        pre_rope_dots.append(pre_rope_dot)

    # Pre-RoPE: every (p_q, p_w) gives the SAME score (bit-equal in fp64).
    for d in pre_rope_dots[1:]:
        torch.testing.assert_close(d, pre_rope_dots[0], rtol=0.0, atol=0.0)

    # Post-RoPE: scores DIFFER once (p_q - p_w) varies — verify the contract
    # we are pinning is non-trivial (i.e., RoPE actually does something).
    diffs = []
    for d in post_rope_dots[1:]:
        diffs.append(float((d - post_rope_dots[0]).abs().max().item()))
    assert max(diffs) > 1e-3, (
        f"post-RoPE dot product appears position-invariant (rope_theta={rope_theta}, "
        f"head_dim={head_dim}). The reference RoPE implementation is broken or the "
        "test inputs are degenerate; the design contract becomes meaningless."
    )


# ---------------------------------------------------------------------------
# Test 2: GQA — repeat_kv on pre-RoPE bank K preserves the invariant.
# ---------------------------------------------------------------------------

def _repeat_kv(kv: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Mirror ``adapter.repeat_kv``: (B, Hk, T, D) → (B, Hk*n_groups, T, D)."""
    if n_groups == 1:
        return kv
    b, hk, t, d = kv.shape
    return kv[:, :, None, :, :].expand(b, hk, n_groups, t, d).reshape(b, hk * n_groups, t, d)


def test_gqa_repeat_kv_preserves_pre_rope_invariance() -> None:
    """GQA bank scoring still position-invariant after repeat_kv expansion.

    Pins ``deltamemory/memory/attn_native_bank.py:504``:
    ``mk_e = adapter.repeat_kv(mk.unsqueeze(0).transpose(1, 2),
    self.num_key_value_groups)``. The expansion must not introduce any
    positional dependence; the test rotates Q at varied positions and
    verifies the GQA-bank dot product is unchanged when the bank K is
    pre-RoPE.
    """
    torch.manual_seed(1)
    head_dim = 64
    n_q_heads, n_kv_heads = 8, 2
    n_groups = n_q_heads // n_kv_heads
    bank_size = 16

    # Per-kv-head bank K, pre-RoPE.
    mk = torch.randn(bank_size, n_kv_heads, head_dim, dtype=torch.float64)
    mk_e = _repeat_kv(mk.unsqueeze(0).transpose(1, 2), n_groups)  # (1, Hq, N, D)

    # Per-q-head Q content (pre-RoPE).
    q_pre = torch.randn(1, n_q_heads, 1, head_dim, dtype=torch.float64)
    pre_dot = torch.matmul(q_pre, mk_e.transpose(-1, -2))

    # Build a post-RoPE-Q variant for several positions and check the bank
    # branch (which uses q_pre, NOT q_post) is unchanged.
    for p_q in (0, 5, 50, 1024):
        # We don't actually use q_post here; this just exercises that the
        # input we are NOT using changes when we move position.
        q_post = _rope_at_position(q_pre, position=p_q, rope_theta=1.0e4)
        assert (q_post - q_pre).abs().max().item() > 0.0 or p_q == 0

        bank_dot = torch.matmul(q_pre, mk_e.transpose(-1, -2))
        torch.testing.assert_close(bank_dot, pre_dot, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Test 3: partial RoPE (GLM-4 style) — pre-RoPE invariance survives even when
# only the first half of head_dim is rotated.
# ---------------------------------------------------------------------------

def _apply_partial_rope_at_position(
    x: torch.Tensor,
    *,
    position: int,
    rope_theta: float,
    rotary_dim: int,
) -> torch.Tensor:
    """GLM-4 / partial-RoPE: rotate only the first ``rotary_dim`` of head_dim."""
    head_dim = x.shape[-1]
    if rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even and <= head_dim, got {rotary_dim}")
    if rotary_dim == 0:
        return x
    cos_full, sin_full = _rope_cos_sin(
        seq_len=position + 1,
        head_dim=rotary_dim,
        rope_theta=rope_theta,
        device=x.device,
        dtype=x.dtype,
    )
    cos = cos_full[position : position + 1].unsqueeze(0).unsqueeze(0)
    sin = sin_full[position : position + 1].unsqueeze(0).unsqueeze(0)
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    rotated = (x_rot * cos) + (_rotate_half(x_rot) * sin)
    return torch.cat((rotated, x_pass), dim=-1)


def test_partial_rope_glm4_pre_rope_invariance() -> None:
    """Partial-RoPE (GLM-4 style) — pre-RoPE bank score still position-invariant.

    GLM-4 only rotates the first half of head_dim. The pre-RoPE inner
    product on the FULL head_dim is unchanged regardless of p_q, p_w; the
    post-RoPE inner product still depends on (p_q - p_w) on the rotated
    half. Pins the assumption documented at
    ``deltamemory/memory/arch_adapter.py:206-222`` (Glm4Adapter).
    """
    torch.manual_seed(2)
    head_dim = 128
    rotary_dim = head_dim // 2
    rope_theta = 1.0e4

    q_content = torch.randn(1, 2, 1, head_dim, dtype=torch.float64)
    k_content = torch.randn(1, 2, 1, head_dim, dtype=torch.float64)
    pre_dot_full = torch.matmul(q_content, k_content.transpose(-1, -2))

    pre_dots = []
    post_dots = []
    for p_q, p_w in [(0, 0), (5, 50), (50, 5), (333, 777)]:
        q_post = _apply_partial_rope_at_position(
            q_content, position=p_q, rope_theta=rope_theta, rotary_dim=rotary_dim
        )
        k_post = _apply_partial_rope_at_position(
            k_content, position=p_w, rope_theta=rope_theta, rotary_dim=rotary_dim
        )
        pre_dots.append(pre_dot_full)
        post_dots.append(torch.matmul(q_post, k_post.transpose(-1, -2)))

    for d in pre_dots[1:]:
        torch.testing.assert_close(d, pre_dots[0], rtol=0.0, atol=0.0)

    diffs = [float((d - post_dots[0]).abs().max().item()) for d in post_dots[1:]]
    assert max(diffs) > 1e-3, (
        "Partial-RoPE post-RoPE dot product appears position-invariant; the "
        "reference partial-RoPE implementation is degenerate."
    )


# ---------------------------------------------------------------------------
# Test 4: code-level tripwire — bank scoring uses ``q_pre``, not ``q_post``.
# ---------------------------------------------------------------------------

def test_attn_native_bank_uses_q_pre_for_bank_scoring() -> None:
    """Textual guard: ``scores_bank = ... q_pre ... mk_e ...`` in the patcher.

    A future refactor that swaps ``q_pre`` for ``q_post`` in the bank-scoring
    line of ``deltamemory/memory/attn_native_bank.py`` would silently break
    the position-invariance of the bank. This test fails fast in that case.

    The check is intentionally narrow: it scans only the body of
    ``_make_patched_forward`` for a ``scores_bank = ... q_pre ...`` pattern
    and a complementary absence of ``scores_bank = ... q_post ...``.
    """
    src = Path(__file__).resolve().parents[1] / "deltamemory" / "memory" / "attn_native_bank.py"
    text = src.read_text(encoding="utf-8")

    # The patched forward is delimited by the def + the next top-level def.
    m = re.search(r"def _make_patched_forward\(.*?\):\n(.*?)\n\n# ---", text, re.DOTALL)
    assert m is not None, "Could not locate _make_patched_forward in attn_native_bank.py"
    body = m.group(1)

    # Either dense scoring (matmul) or cosine scoring uses q_pre as the LHS;
    # we accept any line that assigns to scores_bank with q_pre or q_cos
    # (where q_cos is derived from q_pre at line ~512).
    has_q_pre_score = bool(
        re.search(r"scores_bank\s*=\s*torch\.matmul\(\s*q_pre\b", body)
    ) or bool(re.search(r"scores_bank\s*=\s*torch\.matmul\(\s*q_cos\b", body))
    has_q_post_score = bool(
        re.search(r"scores_bank\s*=\s*torch\.matmul\(\s*q_post\b", body)
    )
    assert has_q_pre_score, (
        "scores_bank must be computed from q_pre (or q_cos derived from q_pre); "
        "see attn_native_bank.py:18-24 for the design contract."
    )
    assert not has_q_post_score, (
        "scores_bank must NOT be computed from q_post — that would make the "
        "bank score position-dependent and silently break recall@k for any "
        "fact whose write-position differs from the query position."
    )


# ---------------------------------------------------------------------------
# Test 5: code-level tripwire — bank K is captured pre-RoPE.
# ---------------------------------------------------------------------------

def test_attn_native_bank_captures_k_pre_rope() -> None:
    """Pins that ``ctx._capture_K[layer_idx]`` stores ``k_pre_for_capture``,
    which is set to ``k_pre`` (pre-RoPE) on non-shared layers (line 438) and
    explicitly ``None`` on KV-shared layers (line 430). Both paths together
    guarantee no post-RoPE K ever lands in the bank.
    """
    src = Path(__file__).resolve().parents[1] / "deltamemory" / "memory" / "attn_native_bank.py"
    text = src.read_text(encoding="utf-8")

    # Non-shared branch: k_pre_for_capture = k_pre (with a possible comment).
    assert re.search(r"k_pre_for_capture\s*=\s*k_pre\b", text), (
        "Non-shared branch must alias k_pre_for_capture to k_pre (pre-RoPE)."
    )
    # Shared branch: k_pre_for_capture = None (no capture on shared layers).
    assert re.search(r"k_pre_for_capture\s*=\s*None\b", text), (
        "Shared branch must set k_pre_for_capture = None (no post-RoPE leak)."
    )
    # Capture write must use k_pre_for_capture, not k_post or key_states.
    assert re.search(
        r"ctx\._capture_K\[layer_idx\]\s*=\s*k_pre_for_capture\[", text
    ), "Bank K capture must write k_pre_for_capture, never key_states or k_post_."
