"""Mneme vLLM integration — AttnNativeBank via model forward-hook.

Import guard: vLLM is an **optional** runtime dependency.  All public symbols
are importable without vLLM installed; the ``BankAttachedLLM`` class raises
``ImportError`` at instantiation time if vLLM is missing.
"""
from __future__ import annotations

from integrations.vllm.bank_attached_llm import BankAttachedLLM  # noqa: F401

__all__ = ["BankAttachedLLM"]
