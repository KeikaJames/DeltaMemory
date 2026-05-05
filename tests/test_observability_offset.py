"""Regression tests for the exporter offset-retry semantics (PR #30 P2).

Both :class:`PrometheusExporter` and :class:`OTelExporter` advance their
internal ``_offset`` cursor **only** after ``parse_records`` succeeds. If
parsing raises (malformed / unknown row), the cursor stays put so the same
batch is retried on the next flush. This prevents silent data loss of valid
observations queued alongside a single bad row.
"""

from __future__ import annotations

import pytest


pytest.importorskip("prometheus_client")


class _StubRecorder:
    def __init__(self, records):
        self._records = list(records)


def _good(signal_name: str, *, value: float = 1.0):
    return {
        "step": 0,
        "layer": 0,
        "token": -1,
        "signal_name": signal_name,
        "value": value,
    }


def _malformed_lopi():
    # signal name has the lopi_ prefix so it passes the canonical filter,
    # but is not registered → parse_records raises ValueError.
    return _good("lopi_unknown_signal")


def test_prometheus_offset_does_not_advance_on_parse_failure():
    from deltamemory.observability.prometheus import PrometheusExporter

    rec = _StubRecorder([_good("scar_proj_mass"), _malformed_lopi()])
    exp = PrometheusExporter(recorder=rec)

    with pytest.raises(ValueError):
        exp.flush()

    # cursor stayed at 0 → on retry (after the bad row is removed by the
    # producer) the valid record is still exported, not silently lost.
    assert exp._offset == 0

    rec._records = [_good("scar_proj_mass")]
    n = exp.flush()
    assert n == 1
    assert exp._offset == 1


def test_otel_offset_does_not_advance_on_parse_failure():
    pytest.importorskip("opentelemetry")
    from deltamemory.observability.otel import OTelExporter

    rec = _StubRecorder([_good("scar_proj_mass"), _malformed_lopi()])
    exp = OTelExporter(recorder=rec)

    with pytest.raises(ValueError):
        exp.flush()

    assert exp._offset == 0

    rec._records = [_good("scar_proj_mass")]
    n = exp.flush()
    assert n == 1
    assert exp._offset == 1
