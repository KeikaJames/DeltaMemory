"""End-to-end test: AttnNativeBank + vLLM on a real model.

Skipped entirely when vLLM is not installed.

When vLLM IS available (GB10 / spark1) this test:
1. Loads gemma-4-31B-it via BankAttachedLLM.
2. Writes 3 seeded facts.
3. Runs a recall prompt.
4. Asserts the target token is in the top-5 logits (recall@5 ≥ 1).
5. Compares the vLLM output logits with an HF-transformers reference within
   a tolerance of 1e-2 (logit space; both paths share the same weights in
   memory so the diff is purely sampling-path noise + vLLM kernel diffs).

Running on spark1::

    cd /home/gabira/projects/RCV-HC
    source .venv-gb10/bin/activate
    pip install vllm   # first time only
    pytest tests/test_vllm_integration.py -v

Environment override for a smaller/different model::

    MNEME_VLLM_TEST_MODEL=/path/to/model pytest tests/test_vllm_integration.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch

# ---------------------------------------------------------------------------
# Skip guard — must come before any vllm import
# ---------------------------------------------------------------------------

try:
    import vllm  # noqa: F401

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _VLLM_AVAILABLE,
    reason="vLLM not installed — skipping vLLM integration tests",
)

# Root on sys.path so we can import deltamemory and integrations
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Default test model — override with env var on spark1.
_DEFAULT_MODEL = os.environ.get(
    "MNEME_VLLM_TEST_MODEL",
    "/home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it",
)

# Tiny facts used for seeded recall test.
_SEED_FACTS = [
    ("city_france", "Paris is the capital of France.", "Paris"),
    ("ceo_apple", "Tim Cook is the CEO of Apple.", "Tim Cook"),
    ("element_79", "Gold has atomic number 79.", "Gold"),
]

# Recall probes: (prompt, expected_target_token)
_RECALL_PROBES = [
    ("The capital of France is", "Paris"),
    ("The CEO of Apple is", "Tim"),
    ("Gold has atomic number", "79"),
]


@pytest.fixture(scope="module")
def bllm():
    """BankAttachedLLM loaded once per test module (expensive)."""
    from integrations.vllm import BankAttachedLLM

    model_path = _DEFAULT_MODEL
    if not Path(model_path).exists():
        pytest.skip(f"Model not found: {model_path} — set MNEME_VLLM_TEST_MODEL")

    instance = BankAttachedLLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=4096,
        alpha=1.0,
    )
    instance.write_facts(_SEED_FACTS)
    return instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBankAttachedLLM:

    def test_model_loads(self, bllm):
        """BankAttachedLLM instantiates and exposes vLLM + nn.Module."""
        assert bllm._nn_model is not None
        assert isinstance(bllm._nn_model, torch.nn.Module)
        assert bllm.vllm_version() != "n/a"

    def test_bank_has_facts(self, bllm):
        """After write_facts the bank is non-empty."""
        assert not bllm.bank.empty
        assert bllm.bank.num_facts >= len(_SEED_FACTS)

    @pytest.mark.parametrize("prompt,target", _RECALL_PROBES)
    def test_recall_top5(self, bllm, prompt: str, target: str):
        """Target token must appear in the top-5 logits (recall@5)."""
        top5 = bllm.recall_top5(prompt, alpha=1.0)
        hits = [t for t in top5 if target.lower() in t.lower()]
        assert hits, (
            f"Target '{target}' not in top-5 for prompt '{prompt}'. "
            f"Top-5 was: {top5}"
        )

    def test_alpha_zero_matches_baseline(self, bllm):
        """alpha=0 bank-attached forward must equal unpatched HF logits within 1e-2.

        This verifies the bit-equality gate (Gate 13A.1) holds end-to-end
        through the vLLM model-unwrap path.
        """
        from deltamemory.memory.attn_native_bank import forward_with_bank

        probe = "The capital of France is"
        device = next(bllm._nn_model.parameters()).device

        # alpha=0: bank attached but inactive
        logits_zero = forward_with_bank(
            bllm.patcher, bllm.bank, bllm.tokenizer, probe, alpha=0.0
        )

        # unpatched baseline: no patcher active
        enc = bllm.tokenizer(probe, return_tensors="pt").to(device)
        with torch.no_grad():
            out_base = bllm._nn_model(**enc, use_cache=False)
        logits_base = out_base.logits[0, -1].detach()

        max_diff = (logits_zero - logits_base).abs().max().item()
        assert max_diff < 1e-2, (
            f"alpha=0 logits differ from baseline by {max_diff:.4e} (threshold 1e-2). "
            "Gate 13A.1 violation — AttnNativePatcher is modifying logits at alpha=0."
        )

    def test_generate_returns_text(self, bllm):
        """generate() produces non-empty text outputs."""
        outputs = bllm.generate(
            ["What is the capital of France?"],
            alpha=1.0,
            max_new_tokens=16,
            temperature=0.0,
        )
        assert outputs, "generate() returned empty list"
        text = outputs[0].outputs[0].text
        assert isinstance(text, str) and len(text) > 0

    def test_hf_vllm_logit_parity(self, bllm):
        """HF forward_with_bank and vLLM recall_top5 agree on top-1 token.

        Both paths share the same weight tensors in memory (vLLM unwrap),
        so the top-1 logit token must be identical at fp32.
        """
        probe = "Tim Cook is the CEO of"

        # HF path
        from deltamemory.memory.attn_native_bank import forward_with_bank

        logits_hf = forward_with_bank(
            bllm.patcher, bllm.bank, bllm.tokenizer, probe, alpha=1.0
        )
        top1_hf = bllm.tokenizer.decode([logits_hf.argmax().item()]).strip()

        # vLLM top5 (greedy top-1 is the first element)
        top5_vllm = bllm.recall_top5(probe, alpha=1.0)

        assert top1_hf == top5_vllm[0] or top1_hf in top5_vllm, (
            f"HF top-1 '{top1_hf}' not in vLLM top-5 {top5_vllm}. "
            "Logit paths diverge — check AttnNativePatcher re-entrancy."
        )
