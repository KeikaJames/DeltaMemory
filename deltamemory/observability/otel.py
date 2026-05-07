"""OpenTelemetry tracing instrumentation for Mneme bank operations."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from deltamemory.diagnostics_schema import SIGNAL_REGISTRY, parse_records


def _load_otel_modules():
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "OpenTelemetry tracing requires opentelemetry-api and opentelemetry-sdk. "
            "Install with `pip install opentelemetry-api opentelemetry-sdk`."
        ) from exc
    return metrics, trace, SERVICE_NAME, Resource, TracerProvider, BatchSpanProcessor, SimpleSpanProcessor


def setup_otel(service_name: str = "mneme", endpoint: str | None = None):
    """Setup OpenTelemetry with tracing exporter.

    Args:
        service_name: Name of the service for tracing.
        endpoint: Optional OTLP endpoint. If None, uses console exporter.

    Returns:
        Configured TracerProvider.
    """
    (
        metrics,
        trace,
        SERVICE_NAME,
        Resource,
        TracerProvider,
        BatchSpanProcessor,
        SimpleSpanProcessor,
    ) = _load_otel_modules()

    resource = Resource(attributes={SERVICE_NAME: service_name})
    tracer_provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=endpoint)
        except ImportError:
            raise ImportError(
                "OTLP exporter requires opentelemetry-exporter-otlp. "
                "Install with `pip install opentelemetry-exporter-otlp`."
            )
    else:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter
        exporter = ConsoleSpanExporter()
        span_processor = SimpleSpanProcessor(exporter)

    if endpoint:
        span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(tracer_provider)
    return tracer_provider


class BankTracer:
    """Trace Mneme bank operations with OpenTelemetry."""

    def __init__(self, tracer_name: str = "deltamemory.bank", layer: int | str = 0) -> None:
        """Initialize tracer.

        Args:
            tracer_name: Name for the tracer.
            layer: Layer identifier for span attributes.
        """
        try:
            from opentelemetry import trace
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetry tracing requires opentelemetry-api. "
                "Install with `pip install opentelemetry-api opentelemetry-sdk`."
            ) from exc

        self.tracer = trace.get_tracer(tracer_name)
        self.layer = layer

    @contextmanager
    def forward_with_bank_span(self, context: str = "") -> Generator[Any, None, None]:
        """Span for forward_with_bank call.

        Args:
            context: Optional context description.

        Yields:
            The active span for adding events.
        """
        span_name = "mneme.forward_with_bank"
        if context:
            span_name = f"{span_name}[{context}]"

        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("layer", self.layer)
            span.set_attribute("timestamp", time.time())
            yield span

    @contextmanager
    def write_fact_span(self, fact_id: str = "") -> Generator[Any, None, None]:
        """Span for write_fact call.

        Args:
            fact_id: Optional fact identifier.

        Yields:
            The active span for adding events.
        """
        span_name = "mneme.write_fact"
        if fact_id:
            span_name = f"{span_name}[{fact_id}]"

        with self.tracer.start_as_current_span(span_name) as span:
            span.set_attribute("layer", self.layer)
            span.set_attribute("timestamp", time.time())
            yield span

    def record_eviction_event(self, span: Any, count: int = 1, reason: str = "capacity") -> None:
        """Record an eviction event on the given span.

        Args:
            span: The span to record the event on.
            count: Number of items evicted.
            reason: Reason for eviction.
        """
        span.add_event(
            "bank.eviction",
            attributes={"count": count, "reason": reason, "layer": self.layer},
        )

    def record_hit_event(self, span: Any, fact_id: str = "") -> None:
        """Record a memory hit event.

        Args:
            span: The span to record on.
            fact_id: Optional fact identifier.
        """
        span.add_event(
            "bank.hit",
            attributes={"fact_id": fact_id, "layer": self.layer},
        )

    def record_miss_event(self, span: Any, fact_id: str = "") -> None:
        """Record a memory miss event.

        Args:
            span: The span to record on.
            fact_id: Optional fact identifier.
        """
        span.add_event(
            "bank.miss",
            attributes={"fact_id": fact_id, "layer": self.layer},
        )


class OTelExporter:
    """Record each canonical injector diagnostic signal as an OTel Histogram."""

    def __init__(self, recorder: Any, meter_provider: Any = None) -> None:
        self.recorder = recorder
        self._offset = 0
        try:
            from opentelemetry import metrics
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "OpenTelemetry diagnostics export requires opentelemetry-api. "
                "Install it with `pip install opentelemetry-api`."
            ) from exc

        if meter_provider is None:
            meter = metrics.get_meter("deltamemory.diagnostics")
        else:
            meter = meter_provider.get_meter("deltamemory.diagnostics")
        self._histograms = {}
        for injector, names in SIGNAL_REGISTRY.items():
            for signal_name in sorted(names):
                self._histograms[(injector, signal_name)] = meter.create_histogram(
                    f"mneme.{injector}.{signal_name}",
                    unit="1",
                    description=f"Mneme {injector.upper()} diagnostic signal {signal_name}",
                )

    def flush(self) -> int:
        """Record observations added to the recorder since the previous flush.

        Offset is advanced ONLY after ``parse_records`` succeeds. If parsing
        raises (e.g. on a malformed/unknown row), the cursor stays put so the
        same batch is retried on the next flush — preventing silent data loss
        of the valid observations queued alongside the bad row.
        """

        records = getattr(self.recorder, "_records", [])
        new_records = records[self._offset :]
        canonical = [
            rec for rec in new_records
            if isinstance(rec, dict)
            and any(str(rec.get("signal_name", "")).startswith(f"{inj}_") for inj in SIGNAL_REGISTRY)
        ]
        signals = parse_records(canonical)
        self._offset = len(records)
        for sig in signals:
            self._histograms[(sig.injector, sig.signal_name)].record(
                sig.value,
                attributes={"layer": sig.layer},
            )
        return len(signals)


__all__ = ["setup_otel", "BankTracer", "OTelExporter"]
