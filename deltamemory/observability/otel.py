"""OpenTelemetry metrics exporter for Mneme injector diagnostics."""
from __future__ import annotations

from typing import Any

from deltamemory.diagnostics_schema import SIGNAL_REGISTRY, parse_records


def _load_otel_metrics():
    try:
        from opentelemetry import metrics
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "OpenTelemetry diagnostics export requires opentelemetry-api. "
            "Install it with `pip install opentelemetry-api`."
        ) from exc
    return metrics


class OTelExporter:
    """Record each canonical injector diagnostic signal as an OTel Histogram."""

    def __init__(self, recorder: Any, meter_provider: Any = None) -> None:
        self.recorder = recorder
        self._offset = 0
        metrics = _load_otel_metrics()
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


__all__ = ["OTelExporter"]
