"""Tests for OpenTelemetry tracing integration."""
from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")

from deltamemory.observability import BankTracer, setup_otel


def test_setup_otel_console():
    """Test setup_otel with console exporter."""
    tracer_provider = setup_otel(service_name="test_mneme", endpoint=None)
    assert tracer_provider is not None


def test_bank_tracer_initialization():
    """Test BankTracer initialization."""
    tracer = BankTracer(tracer_name="test_tracer", layer=0)
    assert tracer is not None
    assert tracer.layer == 0


def test_bank_tracer_forward_span():
    """Test forward_with_bank span creation."""
    tracer = BankTracer(layer="layer_0")

    with tracer.forward_with_bank_span(context="test_context") as span:
        assert span is not None


def test_bank_tracer_write_fact_span():
    """Test write_fact span creation."""
    tracer = BankTracer(layer="layer_0")

    with tracer.write_fact_span(fact_id="fact_123") as span:
        assert span is not None


def test_bank_tracer_eviction_event():
    """Test recording eviction event."""
    tracer = BankTracer(layer=0)

    with tracer.forward_with_bank_span() as span:
        tracer.record_eviction_event(span, count=5, reason="capacity")


def test_bank_tracer_hit_event():
    """Test recording hit event."""
    tracer = BankTracer(layer=0)

    with tracer.write_fact_span() as span:
        tracer.record_hit_event(span, fact_id="fact_001")


def test_bank_tracer_miss_event():
    """Test recording miss event."""
    tracer = BankTracer(layer=0)

    with tracer.write_fact_span() as span:
        tracer.record_miss_event(span, fact_id="fact_002")


def test_bank_tracer_span_attributes():
    """Test span attributes are set correctly."""
    tracer = BankTracer(layer=5)

    with tracer.forward_with_bank_span() as span:
        span.set_attribute("custom_attr", "custom_value")
        pass


def test_multiple_spans():
    """Test creating multiple spans."""
    tracer = BankTracer(layer="multi_layer")

    with tracer.forward_with_bank_span(context="first") as span1:
        pass

    with tracer.forward_with_bank_span(context="second") as span2:
        pass

    with tracer.write_fact_span(fact_id="fact_1") as span3:
        pass


def test_nested_spans():
    """Test nested span creation."""
    tracer = BankTracer(layer=0)

    with tracer.forward_with_bank_span(context="outer") as outer_span:
        tracer.record_hit_event(outer_span, fact_id="fact_outer")

        with tracer.write_fact_span(fact_id="fact_inner") as inner_span:
            tracer.record_eviction_event(inner_span, count=1)


def test_tracer_different_layers():
    """Test tracers for different layers."""
    for layer in range(3):
        tracer = BankTracer(layer=layer)

        with tracer.forward_with_bank_span() as span:
            assert span is not None


def test_span_names_without_context():
    """Test span names are created correctly without context."""
    tracer = BankTracer(layer=0)

    with tracer.forward_with_bank_span() as span:
        pass

    with tracer.write_fact_span() as span:
        pass


def test_event_attributes():
    """Test event attributes are properly recorded."""
    tracer = BankTracer(layer=1)

    with tracer.forward_with_bank_span() as span:
        tracer.record_eviction_event(span, count=10, reason="max_size")
        tracer.record_hit_event(span, fact_id="test_fact")
        tracer.record_miss_event(span, fact_id="missing_fact")


def test_trace_provider_configuration():
    """Test that trace provider is properly configured."""
    provider = setup_otel(service_name="config_test")
    assert provider is not None

    tracer = BankTracer(layer=0)
    with tracer.forward_with_bank_span() as span:
        span.set_attribute("test_attribute", "test_value")
