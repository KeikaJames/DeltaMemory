"""Typed schema for Mneme injector diagnostic signals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


InjectorName = Literal["lopi", "scar", "caa"]


@dataclass(frozen=True)
class InjectorDiagSignal:
    """One long-format diagnostic observation emitted by an injector."""

    step: int
    layer: int
    token: int
    injector: InjectorName
    signal_name: str
    value: float


SIGNAL_REGISTRY: dict[str, set[str]] = {
    "lopi": {"lopi_gamma_t", "lopi_w_ell", "lopi_m_perp_energy_ratio"},
    "scar": {"scar_proj_mass", "scar_ortho_residue", "scar_alpha_drift"},
    "caa": {"caa_steer_norm", "caa_gate_mean", "caa_hidden_drift_ratio"},
}


def _infer_injector(signal_name: str) -> InjectorName:
    for injector in ("lopi", "scar", "caa"):
        if signal_name.startswith(f"{injector}_"):
            return injector
    raise ValueError(
        f"unknown injector diagnostic signal {signal_name!r}; expected prefix "
        "'lopi_', 'scar_', or 'caa_'"
    )


def parse_records(records: list[dict]) -> list[InjectorDiagSignal]:
    """Convert recorder dictionaries into validated typed injector signals."""

    signals: list[InjectorDiagSignal] = []
    for i, rec in enumerate(records):
        try:
            signal_name = str(rec["signal_name"])
            injector = _infer_injector(signal_name)
            if signal_name not in SIGNAL_REGISTRY[injector]:
                raise ValueError(
                    f"signal {signal_name!r} is not registered for injector {injector!r}"
                )
            signals.append(
                InjectorDiagSignal(
                    step=int(rec["step"]),
                    layer=int(rec["layer"]),
                    token=int(rec["token"]),
                    injector=injector,
                    signal_name=signal_name,
                    value=float(rec["value"]),
                )
            )
        except KeyError as exc:
            raise ValueError(f"record {i} missing required field {exc.args[0]!r}") from exc
    return signals


__all__ = ["InjectorDiagSignal", "SIGNAL_REGISTRY", "parse_records"]
