"""MnemeLogitsProcessor — vLLM-compatible logits processor stub.

Satisfies the vLLM ``LogitsProcessor`` callable contract::

    __call__(token_ids: List[int], logits: torch.Tensor) -> torch.Tensor

No vLLM dependency at import time; vLLM is an optional runtime dependency.
The real CAA / SCAR delta functions are wired in via ``set_delta_fn()`` in
V.2 / V.3 of the integration plan.

Alpha=0 bit-equality guarantee
-------------------------------
When ``alpha=0`` the return value is *bit-identical* to the input ``logits``
tensor (the same object is returned; no copy, no mutation).
"""
from __future__ import annotations

from typing import Callable, List, Optional

import torch


class MnemeLogitsProcessor:
    """Logits processor compatible with vLLM's ``SamplingParams(logits_processors=[...])``.

    Parameters
    ----------
    injector:
        Any callable with signature
        ``(token_ids: list[int], logits: torch.Tensor) -> torch.Tensor``,
        OR a ``deltamemory.memory.caa_injector.CAAInjector`` instance.
        In the stub phase this is accepted but not invoked directly — use
        ``set_delta_fn`` to register the active delta function instead.
    alpha:
        Scaling coefficient for the residual delta.  ``alpha=0`` bypasses
        all computation and returns ``logits`` unchanged (bit-equal).
    """

    def __init__(
        self,
        injector: Optional[object] = None,
        *,
        alpha: float = 1.0,
    ) -> None:
        self.injector = injector
        self.alpha = alpha
        self._delta_fn: Optional[Callable[[List[int], torch.Tensor], torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_delta_fn(
        self, fn: Callable[[List[int], torch.Tensor], torch.Tensor]
    ) -> None:
        """Register a user-supplied delta function.

        The function must have the signature::

            fn(token_ids: list[int], logits: torch.Tensor) -> torch.Tensor

        It is called at each decoding step when ``alpha != 0``.
        """
        self._delta_fn = fn

    def reset(self) -> None:
        """Clear any cached or stateful data (e.g. KV-state for future SCAR use)."""
        self._delta_fn = None

    # ------------------------------------------------------------------
    # vLLM LogitsProcessor contract
    # ------------------------------------------------------------------

    def __call__(
        self, token_ids: List[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """Apply the residual delta to ``logits``.

        When ``alpha=0`` returns ``logits`` unchanged (same object, bit-equal).
        When ``alpha>0`` returns ``logits + alpha * delta`` where ``delta``
        comes from ``set_delta_fn`` (or is a zero tensor if none is set).
        """
        if self.alpha == 0.0:
            return logits

        delta = self._inj_delta(token_ids, logits)
        return logits + self.alpha * delta

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inj_delta(
        self, token_ids: List[int], logits: torch.Tensor
    ) -> torch.Tensor:
        """Return the residual delta tensor.

        Uses the registered ``_delta_fn`` if present; otherwise returns a
        zero tensor of the same shape / dtype / device as ``logits``.
        """
        if self._delta_fn is not None:
            return self._delta_fn(token_ids, logits)
        return torch.zeros_like(logits)
