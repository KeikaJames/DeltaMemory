"""Tests for MnemeLogitsProcessor (V.1 stub).

All tests run on CPU with tiny tensors — no GPU, no vLLM required.
"""
from __future__ import annotations

import importlib
import sys

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logits(n: int = 8, seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(n, generator=g)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMnemeLogitsProcessor:

    def test_alpha_zero_bit_equality(self):
        """alpha=0 must return the *same object* (bit-equal, no copy)."""
        from examples.vllm_integration.mneme_vllm.logits_processor import MnemeLogitsProcessor

        logits = _make_logits()
        mp = MnemeLogitsProcessor(alpha=0.0)
        out = mp(list(range(4)), logits)

        assert out is logits, "alpha=0 must return the original tensor unchanged"

    def test_alpha_nonzero_default_delta_is_zero(self):
        """alpha>0 with no delta_fn registered: output equals logits (zero delta)."""
        from examples.vllm_integration.mneme_vllm.logits_processor import MnemeLogitsProcessor

        logits = _make_logits()
        mp = MnemeLogitsProcessor(alpha=2.0)
        out = mp(list(range(4)), logits)

        assert torch.allclose(out, logits, atol=0.0), (
            "Default (zero) delta should leave logits unchanged even with alpha>0"
        )

    def test_alpha_nonzero_custom_delta(self):
        """alpha>0 with a custom delta_fn: output = logits + alpha * delta."""
        from examples.vllm_integration.mneme_vllm.logits_processor import MnemeLogitsProcessor

        logits = _make_logits()
        delta = torch.ones_like(logits) * 0.5
        alpha = 3.0

        mp = MnemeLogitsProcessor(alpha=alpha)
        mp.set_delta_fn(lambda ids, lgt: delta)

        out = mp(list(range(4)), logits)
        expected = logits + alpha * delta

        assert torch.allclose(out, expected, atol=1e-6), (
            "Output should be logits + alpha * delta"
        )

    def test_reset_clears_delta_fn(self):
        """reset() should clear the registered delta function."""
        from examples.vllm_integration.mneme_vllm.logits_processor import MnemeLogitsProcessor

        logits = _make_logits()
        mp = MnemeLogitsProcessor(alpha=1.0)
        mp.set_delta_fn(lambda ids, lgt: torch.ones_like(lgt) * 99.0)

        # Before reset: delta is applied
        out_before = mp([], logits)
        assert not torch.allclose(out_before, logits, atol=0.0)

        mp.reset()

        # After reset: delta_fn is None → zero delta → logits unchanged
        out_after = mp([], logits)
        assert torch.allclose(out_after, logits, atol=0.0), (
            "After reset(), delta_fn should be cleared (zero delta)"
        )

    def test_import_does_not_require_vllm(self):
        """Importing the module must not raise ImportError even without vLLM."""
        # Temporarily hide vllm from sys.modules if it happens to be installed
        vllm_backup = sys.modules.pop("vllm", None)
        try:
            mod = importlib.import_module(
                "examples.vllm_integration.mneme_vllm.logits_processor"
            )
            assert hasattr(mod, "MnemeLogitsProcessor")
        except ImportError as exc:
            pytest.fail(f"Import raised ImportError without vllm installed: {exc}")
        finally:
            if vllm_backup is not None:
                sys.modules["vllm"] = vllm_backup
