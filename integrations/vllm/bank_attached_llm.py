"""BankAttachedLLM — vLLM + AttnNativeBank real integration.

Architecture
------------
vLLM's ``LLM`` class loads the full model weights into GPU memory and exposes
the underlying ``torch.nn.Module`` through the worker's ``model_runner``.
We leverage this to apply the **exact same** ``AttnNativePatcher`` that the
HF-transformers path uses, giving bit-equal write/read semantics across both
inference back-ends.

Integration path (vLLM ≥ 0.4):

    llm = vllm.LLM(model=..., ...)
    underlying_model = _unwrap_vllm_model(llm)   # driver_worker.model_runner.model
    patcher = AttnNativePatcher(underlying_model, adapter)
    bank    = fresh_bank(underlying_model)
    write_fact(patcher, bank, tok, write_prompt, ...)
    outputs = llm.generate([read_prompt], SamplingParams(...))

Because ``AttnNativePatcher.patched()`` / ``.injecting()`` context managers
monkey-patch and restore ``forward`` on the **same** module objects that vLLM
calls, the bank is live for every token of the generation step.

vLLM version note
-----------------
The ``_unwrap_vllm_model`` helper resolves the model access path for vLLM ≥
0.4 (``executor.driver_worker.model_runner.model``).  For older versions or
multi-GPU tensor-parallel runs the path differs; see ``_UNWRAP_PATHS`` below.

No vLLM patching at all
-----------------------
If vLLM is not installed this module still imports cleanly.  Every public
class / function raises ``ImportError`` at call time with a helpful message.
"""
from __future__ import annotations

import contextlib
import sys
from typing import Any, Optional

import torch

# ---------------------------------------------------------------------------
# Soft vLLM import
# ---------------------------------------------------------------------------

try:
    import vllm  # type: ignore
    from vllm import LLM, SamplingParams  # type: ignore

    _VLLM_AVAILABLE = True
    _VLLM_VERSION = getattr(vllm, "__version__", "unknown")
except ImportError:
    _VLLM_AVAILABLE = False
    _VLLM_VERSION = "n/a"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Ordered list of attribute paths tried when unwrapping the underlying model
# from a vLLM LLM instance.  Path depends on vLLM version and executor type.
_UNWRAP_PATHS = [
    # vLLM ≥ 0.5 (GPU executor)
    "llm_engine.model_executor.driver_worker.model_runner.model",
    # vLLM 0.4.x
    "llm_engine.driver_worker.model_runner.model",
    # Ray-based executor (multi-GPU, driver side)
    "llm_engine.model_executor.driver_worker.worker.model_runner.model",
]


def _unwrap_vllm_model(llm: "LLM") -> torch.nn.Module:
    """Extract the raw ``torch.nn.Module`` from a ``vllm.LLM`` instance.

    Tries known attribute paths across vLLM versions.  Raises ``RuntimeError``
    with a diagnostic message if none succeed (pass ``--trust-remote-code`` or
    update the ``_UNWRAP_PATHS`` list for your vLLM version).
    """
    if not _VLLM_AVAILABLE:
        raise ImportError(
            "vLLM is not installed.  Install it with:\n"
            "    pip install vllm\n"
            "then re-run."
        )
    for path in _UNWRAP_PATHS:
        obj: Any = llm
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.Module):
            return obj
    raise RuntimeError(
        f"BankAttachedLLM: could not locate the underlying nn.Module in "
        f"vllm.LLM (version={_VLLM_VERSION}).  "
        f"Tried paths: {_UNWRAP_PATHS}.  "
        "Please update _UNWRAP_PATHS in integrations/vllm/bank_attached_llm.py "
        "for your vLLM version."
    )


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class BankAttachedLLM:
    """vLLM LLM wrapper with an AttnNativeBank attached.

    Usage::

        from integrations.vllm import BankAttachedLLM

        bllm = BankAttachedLLM(
            model="/path/to/gemma-4-31B-it",
            tensor_parallel_size=1,
            dtype="bfloat16",
        )
        bllm.write_facts([
            ("city_capital", "Paris is the capital of France.", "Paris"),
            ("ceo_apple",    "Tim Cook is the CEO of Apple.",   "Tim Cook"),
        ])
        outputs = bllm.generate(
            ["What is the capital of France?"],
            max_new_tokens=32,
            alpha=1.0,
        )
        for o in outputs:
            print(o.text)

    Parameters
    ----------
    model:
        HF model path / identifier forwarded to ``vllm.LLM``.
    alpha:
        Default injection scale.  Can be overridden per ``generate()`` call.
    vllm_kwargs:
        Extra keyword arguments forwarded verbatim to ``vllm.LLM()``.
    """

    def __init__(
        self,
        model: str,
        *,
        alpha: float = 1.0,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        **vllm_kwargs: Any,
    ) -> None:
        if not _VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed.  Install it with:\n"
                "    pip install vllm\n"
                "then retry."
            )

        from deltamemory.memory.attn_native_bank import (
            AttnNativePatcher,
            fresh_bank,
        )
        from transformers import AutoTokenizer  # type: ignore

        self.alpha = alpha
        self.model_id = model

        # --- 1. Launch vLLM ---
        llm_kwargs: dict[str, Any] = dict(
            model=model,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        llm_kwargs.update(vllm_kwargs)
        self.llm: LLM = LLM(**llm_kwargs)

        # --- 2. Unwrap the underlying nn.Module ---
        self._nn_model: torch.nn.Module = _unwrap_vllm_model(self.llm)

        # --- 3. Build patcher + bank using the real model weights ---
        self.patcher = AttnNativePatcher(self._nn_model)
        self.bank = fresh_bank(self._nn_model)

        # --- 4. HF tokenizer for write_fact / address encoding ---
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    # ------------------------------------------------------------------
    # Bank management
    # ------------------------------------------------------------------

    def write_facts(
        self,
        facts: list[tuple[str, str, str]],
        *,
        policy: str = "period",
    ) -> None:
        """Write a list of (fact_id, write_prompt, address) tuples into the bank.

        Parameters
        ----------
        facts:
            Each element is ``(fact_id, write_prompt, address)`` where
            ``write_prompt`` is the full sentence encoding the fact and
            ``address`` is the key token used to locate the injection site.
        policy:
            Capture policy forwarded to ``write_fact()`` (``"period"`` default).
        """
        from deltamemory.memory.attn_native_bank import write_fact as _write_fact

        for fact_id, write_prompt, address in facts:
            _write_fact(
                self.patcher,
                self.bank,
                self.tokenizer,
                write_prompt=write_prompt,
                fact_id=fact_id,
                address=address,
                policy=policy,
            )

    def clear_bank(self) -> None:
        """Remove all facts from the bank (keeps patcher intact)."""
        from deltamemory.memory.attn_native_bank import fresh_bank

        self.bank = fresh_bank(self._nn_model)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _bank_active(self, alpha: float):
        """Context manager: activate bank injection for vLLM forward pass."""
        with self.patcher.patched(), self.patcher.injecting(self.bank, alpha=alpha):
            yield

    def generate(
        self,
        prompts: list[str],
        *,
        alpha: Optional[float] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **sampling_kwargs: Any,
    ) -> list[Any]:
        """Generate text for ``prompts`` with the bank attached.

        Parameters
        ----------
        prompts:
            Plain-text prompt strings.
        alpha:
            Injection scale override (uses ``self.alpha`` if ``None``).
        max_new_tokens:
            ``max_tokens`` forwarded to ``SamplingParams``.
        temperature / top_p:
            Standard sampling parameters.
        sampling_kwargs:
            Extra kwargs forwarded to ``vllm.SamplingParams``.

        Returns
        -------
        list[vllm.RequestOutput]
            One element per prompt.
        """
        _alpha = self.alpha if alpha is None else alpha
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            **sampling_kwargs,
        )
        with self._bank_active(_alpha):
            outputs = self.llm.generate(prompts, params)
        return outputs

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def recall_top5(self, prompt: str, *, alpha: Optional[float] = None) -> list[str]:
        """Return the top-5 token strings at the last position of ``prompt``.

        Useful for recall@5 evaluation without a full decode pass.
        """
        from deltamemory.memory.attn_native_bank import forward_with_bank

        _alpha = self.alpha if alpha is None else alpha
        logits = forward_with_bank(
            self.patcher,
            self.bank,
            self.tokenizer,
            read_prompt=prompt,
            alpha=_alpha,
        )
        top5_ids = logits.topk(5).indices.tolist()
        return [self.tokenizer.decode([tid]).strip() for tid in top5_ids]

    def vllm_version(self) -> str:
        return _VLLM_VERSION

    def __repr__(self) -> str:
        n = self.bank.num_facts if not self.bank.empty else 0
        return (
            f"BankAttachedLLM(model={self.model_id!r}, "
            f"bank_size={n}, alpha={self.alpha}, vllm={_VLLM_VERSION})"
        )
