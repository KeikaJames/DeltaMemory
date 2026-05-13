"""Tests for Prometheus observability integration."""
from __future__ import annotations

import pytest

pytest.importorskip("prometheus_client")

from deltamemory.observability import BankMetrics, InstrumentedBank, wrap_forward_with_bank


def test_bank_metrics_initialization():
    """Test BankMetrics initialization and metric creation."""
    metrics = BankMetrics()
    assert metrics is not None
    assert hasattr(metrics, "bank_size")
    assert hasattr(metrics, "hit_rate_hits")
    assert hasattr(metrics, "hit_rate_accesses")
    assert hasattr(metrics, "alpha")
    assert hasattr(metrics, "forward_latency")
    assert hasattr(metrics, "eviction_total")


def test_bank_metrics_recording():
    """Test recording various metrics."""
    metrics = BankMetrics()

    metrics.set_bank_size(0, 10)
    metrics.set_alpha(0, 0.5)
    metrics.record_access(0)
    metrics.record_hit(0)
    metrics.record_eviction(0)


def test_bank_metrics_latency_timer():
    """Test forward latency timer context manager."""
    import time

    metrics = BankMetrics()
    layer = "layer_0"

    with metrics.forward_latency_timer(layer):
        time.sleep(0.001)

    with metrics.forward_latency_timer(layer):
        time.sleep(0.005)


def test_instrumented_bank_initialization():
    """Test InstrumentedBank wrapper initialization."""

    class MockBank:
        pass

    bank = MockBank()
    metrics = BankMetrics()
    instrumented = InstrumentedBank(bank, metrics, layer=0)

    assert instrumented.bank is bank
    assert instrumented.metrics is metrics
    assert instrumented.layer == 0


def test_instrumented_bank_recording():
    """Test InstrumentedBank recording operations."""

    class MockBank:
        pass

    bank = MockBank()
    instrumented = InstrumentedBank(bank, layer="layer_1")

    instrumented.record_access()
    instrumented.record_hit()
    instrumented.update_bank_size(5)
    instrumented.update_alpha(0.3)
    instrumented.record_eviction()


def test_instrumented_bank_forward_context():
    """Test InstrumentedBank forward_with_bank_instrumented context manager."""

    class MockBank:
        pass

    bank = MockBank()
    instrumented = InstrumentedBank(bank, layer="layer_0")

    with instrumented.forward_with_bank_instrumented():
        pass


def test_wrap_forward_with_bank():
    """Test wrapping a forward function with instrumentation."""

    def mock_forward(*args, **kwargs):
        return "result"

    metrics = BankMetrics()
    wrapped = wrap_forward_with_bank(mock_forward, metrics, layer="layer_0")

    result = wrapped()
    assert result == "result"


def test_wrap_forward_with_bank_with_args():
    """Test wrapped function passes arguments correctly."""

    def mock_forward(x, y, z=10):
        return x + y + z

    wrapped = wrap_forward_with_bank(mock_forward, layer="layer_0")

    result = wrapped(1, 2, z=3)
    assert result == 6


def test_multiple_layers():
    """Test metrics tracking for multiple layers."""
    metrics = BankMetrics()

    for layer in range(5):
        metrics.set_bank_size(layer, 10 * layer)
        metrics.set_alpha(layer, 0.1 * layer)
        metrics.record_access(layer)
        metrics.record_hit(layer)
        metrics.record_eviction(layer)


def test_counter_increments():
    """Test that counters increment correctly."""
    metrics = BankMetrics()
    layer = "test_layer"

    for _ in range(5):
        metrics.record_access(layer)
        metrics.record_hit(layer)
        metrics.record_eviction(layer)


def test_histogram_recording():
    """Test histogram recording with latency timer."""
    import time

    metrics = BankMetrics()
    layer = "latency_test"

    for _ in range(3):
        with metrics.forward_latency_timer(layer):
            time.sleep(0.002)
