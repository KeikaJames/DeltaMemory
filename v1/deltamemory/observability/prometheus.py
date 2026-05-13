"""Prometheus exporter for Mneme bank operations and diagnostics."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from deltamemory.diagnostics_schema import SIGNAL_REGISTRY, parse_records


def _load_prometheus_client():
    try:
        from prometheus_client import Counter, Gauge, Histogram, start_http_server
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Prometheus diagnostics export requires prometheus-client. "
            "Install it with `pip install prometheus-client`."
        ) from exc
    return Counter, Gauge, Histogram, start_http_server


class BankMetrics:
    """Track Mneme memory bank operations and performance.

    This is a singleton to avoid duplicate metric registration in prometheus_client.
    """

    _instance: BankMetrics | None = None

    def __new__(cls) -> BankMetrics:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        Counter, Gauge, Histogram, _ = _load_prometheus_client()

        self.bank_size = Gauge(
            "mneme_bank_size",
            "Number of facts stored in memory bank",
            ["layer"],
        )
        self.hit_rate_hits = Counter(
            "mneme_bank_hit_rate_hits_total",
            "Total number of memory bank hits",
            ["layer"],
        )
        self.hit_rate_accesses = Counter(
            "mneme_bank_hit_rate_accesses_total",
            "Total number of memory bank accesses",
            ["layer"],
        )
        self.alpha = Gauge(
            "mneme_alpha",
            "Memory alpha (mixing coefficient)",
            ["layer"],
        )
        self.forward_latency = Histogram(
            "mneme_forward_latency_seconds",
            "Latency of forward_with_bank call",
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
            labelnames=["layer"],
        )
        self.eviction_total = Counter(
            "mneme_eviction_total",
            "Total number of evictions from memory bank",
            ["layer"],
        )
        self._initialized = True

    @contextmanager
    def forward_latency_timer(self, layer: int | str = "unknown") -> Generator[None, None, None]:
        """Context manager to record forward_with_bank latency."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.forward_latency.labels(layer=str(layer)).observe(elapsed)

    def record_hit(self, layer: int | str = "unknown") -> None:
        """Record a memory bank hit."""
        self.hit_rate_hits.labels(layer=str(layer)).inc()

    def record_access(self, layer: int | str = "unknown") -> None:
        """Record a memory bank access."""
        self.hit_rate_accesses.labels(layer=str(layer)).inc()

    def set_bank_size(self, layer: int | str, size: int) -> None:
        """Set the current bank size."""
        self.bank_size.labels(layer=str(layer)).set(size)

    def set_alpha(self, layer: int | str, alpha: float) -> None:
        """Set the alpha value."""
        self.alpha.labels(layer=str(layer)).set(alpha)

    def record_eviction(self, layer: int | str = "unknown") -> None:
        """Record an eviction event."""
        self.eviction_total.labels(layer=str(layer)).inc()


class PrometheusExporter:
    """Expose one Gauge per canonical injector diagnostic signal."""

    def __init__(self, recorder: Any) -> None:
        self.recorder = recorder
        self._offset = 0
        Counter, Gauge, Histogram, _ = _load_prometheus_client()
        self._gauges = {}
        for injector, names in SIGNAL_REGISTRY.items():
            for signal_name in sorted(names):
                metric_name = f"mneme_{injector}_{signal_name[len(injector) + 1:]}"
                try:
                    self._gauges[(injector, signal_name)] = Gauge(
                        metric_name,
                        f"Mneme {injector.upper()} diagnostic signal {signal_name}",
                        ["layer"],
                    )
                except ValueError:
                    # Metric already registered; retrieve it from registry
                    from prometheus_client import REGISTRY
                    collectors = REGISTRY._collector_to_names
                    for collector in collectors:
                        if hasattr(collector, "_name") and collector._name == metric_name:
                            self._gauges[(injector, signal_name)] = collector
                            break

    def flush(self) -> int:
        """Export recorder rows added since the previous flush.

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
            if (sig.injector, sig.signal_name) in self._gauges:
                self._gauges[(sig.injector, sig.signal_name)].labels(layer=str(sig.layer)).set(sig.value)
        return len(signals)


def serve_http(port: int = 9464):
    """Start a Prometheus scrape endpoint via ``prometheus_client``."""

    _, _, _, start_http_server = _load_prometheus_client()
    return start_http_server(port)


__all__ = ["BankMetrics", "PrometheusExporter", "serve_http"]
