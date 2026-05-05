from __future__ import annotations

import pytest

from deltamemory.diagnostics import DiagnosticRecorder
from deltamemory.diagnostics_schema import InjectorDiagSignal, SIGNAL_REGISTRY, parse_records


def test_signal_registry_round_trip():
    records = []
    for injector, names in SIGNAL_REGISTRY.items():
        for i, name in enumerate(sorted(names)):
            records.append({
                "step": i,
                "layer": i + 1,
                "token": -1,
                "signal_name": name,
                "value": i + 0.25,
            })

    signals = parse_records(records)

    assert len(signals) == sum(len(names) for names in SIGNAL_REGISTRY.values())
    assert {sig.injector for sig in signals} == set(SIGNAL_REGISTRY)
    assert {sig.signal_name for sig in signals} == set().union(*SIGNAL_REGISTRY.values())
    assert all(isinstance(sig, InjectorDiagSignal) for sig in signals)


def test_parse_records_happy_path():
    signals = parse_records([
        {
            "step": "2",
            "layer": "4",
            "token": "-1",
            "signal_name": "scar_proj_mass",
            "value": "0.75",
        }
    ])

    assert signals == [
        InjectorDiagSignal(
            step=2,
            layer=4,
            token=-1,
            injector="scar",
            signal_name="scar_proj_mass",
            value=0.75,
        )
    ]


@pytest.mark.parametrize(
    "record",
    [
        {"step": 0, "layer": 0, "token": -1, "signal_name": "bank_col_sum", "value": 1.0},
        {"step": 0, "layer": 0, "token": -1, "signal_name": "caa_unknown", "value": 1.0},
        {"step": 0, "layer": 0, "token": -1, "value": 1.0},
    ],
)
def test_parse_records_raises_on_unknown_or_malformed(record):
    with pytest.raises(ValueError):
        parse_records([record])


def test_recorder_to_signals_filters_non_injector_rows():
    rec = DiagnosticRecorder(model=None, patcher=None)
    rec._records.extend([
        {"step": 0, "layer": 1, "token": -1, "signal_name": "residual_norm", "value": 3.0},
        {"step": 0, "layer": 1, "token": -1, "signal_name": "caa_gate_mean", "value": 1.0},
    ])

    signals = rec.to_signals()

    assert signals == [
        InjectorDiagSignal(
            step=0,
            layer=1,
            token=-1,
            injector="caa",
            signal_name="caa_gate_mean",
            value=1.0,
        )
    ]
