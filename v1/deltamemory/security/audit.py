"""Audit logging primitives for Mneme security-sensitive operations."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch

_AUDITOR: Optional["AuditLogger"] = None
_EVENT_TYPES = {"inject", "bank_load", "bank_store", "access_denied"}
_EMPTY_HASH = "sha256:" + hashlib.sha256(b"").hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Return the required SHA-256 hash over a tensor's raw contiguous CPU bytes."""
    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    """Return a schema-compatible SHA-256 hash for non-tensor payloads."""
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def set_auditor(auditor: "AuditLogger | None") -> "AuditLogger | None":
    """Install a process-local auditor and return the previous auditor."""
    global _AUDITOR
    prev = _AUDITOR
    _AUDITOR = auditor
    return prev


def get_auditor() -> "AuditLogger | None":
    """Return the currently attached process-local auditor, if any."""
    return _AUDITOR


class AuditLogger:
    """Emit one JSON object per audit event to a file, sink, or in-memory list."""

    def __init__(
        self,
        path: str | None = None,
        sink: Callable[[dict], None] | None = None,
    ) -> None:
        self.path = path
        self.sink = sink
        self.events: list[dict] = []
        self._fh: Any = None
        self._previous: AuditLogger | None = None

    def __enter__(self) -> "AuditLogger":
        self._previous = set_auditor(self)
        return self

    def __exit__(self, *_: Any) -> None:
        set_auditor(self._previous)
        self.close()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def emit(self, event: dict) -> None:
        normalized = normalize_event(event)
        self.events.append(normalized)
        if self.sink is not None:
            self.sink(dict(normalized))
        if self.path is not None:
            if self._fh is None:
                path = Path(self.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = path.open("a", encoding="utf-8")
            self._fh.write(json.dumps(normalized, sort_keys=True) + "\n")
            self._fh.flush()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def _normalize_signal_summary(summary: dict[str, Any] | None) -> dict[str, float | None]:
    summary = summary or {}
    return {
        "steer_norm": _float_or_none(summary.get("steer_norm")),
        "drift_ratio": _float_or_none(summary.get("drift_ratio")),
        "gate_mean": _float_or_none(summary.get("gate_mean")),
    }


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize an event to the public JSON-lines schema."""
    event_type = str(event.get("event_type"))
    if event_type not in _EVENT_TYPES:
        raise ValueError(f"unknown audit event_type: {event_type!r}")

    normalized = {
        "ts_ns": int(event.get("ts_ns", time.time_ns())),
        "event_type": event_type,
        "injector": event.get("injector"),
        "layer": event.get("layer"),
        "alpha": _float_or_none(event.get("alpha")),
        "signal_summary": _normalize_signal_summary(event.get("signal_summary")),
        "vector_hash": event.get("vector_hash") or _EMPTY_HASH,
        "actor": event.get("actor"),
        "request_id": event.get("request_id"),
    }
    for key, value in event.items():
        if key not in normalized and key != "vector_tensor":
            normalized[key] = value
    return normalized


def audit_event(
    *,
    event_type: str,
    injector: str | None = None,
    layer: int | None = None,
    alpha: float | None = None,
    signal_summary: dict[str, Any] | None = None,
    vector_tensor: torch.Tensor | None = None,
    vector_hash: str | None = None,
    actor: str | None = None,
    request_id: str | None = None,
    **extra: Any,
) -> None:
    """Fail-safe global audit hook, mirroring the diagnostics ``_RECORDER`` pattern."""
    try:
        auditor = _AUDITOR
        if auditor is None:
            return
        if vector_hash is None and vector_tensor is not None:
            vector_hash = tensor_sha256(vector_tensor)
        event = {
            "event_type": event_type,
            "injector": injector,
            "layer": layer,
            "alpha": alpha,
            "signal_summary": signal_summary,
            "vector_hash": vector_hash,
            "actor": actor,
            "request_id": request_id,
        }
        event.update(extra)
        auditor.emit(event)
    except Exception:
        pass


__all__ = [
    "AuditLogger",
    "audit_event",
    "bytes_sha256",
    "get_auditor",
    "set_auditor",
    "tensor_sha256",
]
