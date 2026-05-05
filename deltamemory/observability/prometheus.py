"""Prometheus exporter for :class:`deltamemory.diagnostics.DiagnosticRecorder`."""
from __future__ import annotations

from typing import Any

from deltamemory.diagnostics_schema import SIGNAL_REGISTRY, parse_records


def _load_prometheus_client():
    try:
        from prometheus_client import Gauge, start_http_server
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Prometheus diagnostics export requires prometheus_client. "
            "Install it with `pip install prometheus-client`."
        ) from exc
    return Gauge, start_http_server


class PrometheusExporter:
    """Expose one Gauge per canonical injector diagnostic signal."""

    def __init__(self, recorder: Any) -> None:
        self.recorder = recorder
        self._offset = 0
        Gauge, _ = _load_prometheus_client()
        self._gauges = {}
        for injector, names in SIGNAL_REGISTRY.items():
            for signal_name in sorted(names):
                metric_name = f"mneme_{injector}_{signal_name[len(injector) + 1:]}"
                self._gauges[(injector, signal_name)] = Gauge(
                    metric_name,
                    f"Mneme {injector.upper()} diagnostic signal {signal_name}",
                    ["layer"],
                )

    def flush(self) -> int:
        """Export recorder rows added since the previous flush."""

        records = getattr(self.recorder, "_records", [])
        new_records = records[self._offset :]
        self._offset = len(records)
        canonical = [
            rec for rec in new_records
            if isinstance(rec, dict)
            and any(str(rec.get("signal_name", "")).startswith(f"{inj}_") for inj in SIGNAL_REGISTRY)
        ]
        signals = parse_records(canonical)
        for sig in signals:
            self._gauges[(sig.injector, sig.signal_name)].labels(layer=str(sig.layer)).set(sig.value)
        return len(signals)


def serve_http(port: int = 9464):
    """Start a Prometheus scrape endpoint via ``prometheus_client``."""

    _, start_http_server = _load_prometheus_client()
    return start_http_server(port)


__all__ = ["PrometheusExporter", "serve_http"]
