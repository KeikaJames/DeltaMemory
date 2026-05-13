"""Prometheus-instrumented wrappers for Mneme bank operations."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from deltamemory.observability.prometheus import BankMetrics


class InstrumentedBank:
    """Wraps a memory bank with Prometheus instrumentation.

    This class provides a context manager and wrapper methods that instrument
    memory bank operations with Prometheus metrics without modifying the core
    bank implementation.
    """

    def __init__(self, bank: Any, metrics: BankMetrics | None = None, layer: int | str = 0) -> None:
        """Initialize instrumented bank wrapper.

        Args:
            bank: The memory bank object to instrument.
            metrics: BankMetrics instance (created if None).
            layer: Layer identifier for metric labels.
        """
        self.bank = bank
        self.metrics = metrics or BankMetrics()
        self.layer = layer

    @contextmanager
    def forward_with_bank_instrumented(self) -> Generator[None, None, None]:
        """Context manager for forward_with_bank calls with latency tracking."""
        with self.metrics.forward_latency_timer(self.layer):
            yield

    def record_access(self) -> None:
        """Record a memory access."""
        self.metrics.record_access(self.layer)

    def record_hit(self) -> None:
        """Record a memory hit."""
        self.metrics.record_hit(self.layer)

    def update_bank_size(self, size: int) -> None:
        """Update bank size metric."""
        self.metrics.set_bank_size(self.layer, size)

    def update_alpha(self, alpha: float) -> None:
        """Update alpha metric."""
        self.metrics.set_alpha(self.layer, alpha)

    def record_eviction(self) -> None:
        """Record an eviction."""
        self.metrics.record_eviction(self.layer)


def wrap_forward_with_bank(forward_fn: Any, metrics: BankMetrics | None = None, layer: int | str = 0):
    """Wrap forward_with_bank function with instrumentation.

    Args:
        forward_fn: The forward_with_bank function to wrap.
        metrics: BankMetrics instance (created if None).
        layer: Layer identifier for metric labels.

    Returns:
        Wrapped function that records latency and other metrics.
    """
    metrics = metrics or BankMetrics()

    def wrapped(*args, **kwargs):
        with metrics.forward_latency_timer(layer):
            return forward_fn(*args, **kwargs)

    return wrapped


__all__ = ["BankMetrics", "InstrumentedBank", "wrap_forward_with_bank"]
