"""Minimal RBAC stub for bank and injection operations.

Operators must plug in real identity-provider token verification before using
this guard as an authorization boundary.
"""
from __future__ import annotations

from enum import Enum

from deltamemory.security.audit import audit_event
from deltamemory.security.encrypted_bank import BankAuthError


class Role(Enum):
    READER = "reader"
    WRITER = "writer"
    ADMIN = "admin"


Policy = {
    "bank_load": Role.READER,
    "bank_store": Role.WRITER,
    "inject": Role.READER,
    "rotate_key": Role.ADMIN,
}
_RANK = {Role.READER: 1, Role.WRITER: 2, Role.ADMIN: 3}


class AccessGuard:
    """Role hierarchy check only; no token verification is performed."""

    @staticmethod
    def check(operation: str, role: Role, *, actor: str | None = None) -> None:
        try:
            actual = role if isinstance(role, Role) else Role(role)
            required = Policy[operation]
        except Exception as exc:
            audit_event(event_type="access_denied", actor=actor, operation=operation)
            raise BankAuthError(f"unknown operation or role: {operation!r}") from exc

        if _RANK[actual] < _RANK[required]:
            audit_event(
                event_type="access_denied",
                actor=actor,
                operation=operation,
                role=actual.value,
                required_role=required.value,
            )
            raise PermissionError(
                f"operation {operation!r} requires {required.value}; got {actual.value}"
            )


__all__ = ["AccessGuard", "Policy", "Role"]
