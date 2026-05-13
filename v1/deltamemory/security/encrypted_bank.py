"""Fernet-encrypted persistence helpers for tensor-only Mneme banks."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

from deltamemory.security.audit import audit_event, bytes_sha256

try:
    from deltamemory.runtime.errors import InjectorError as _BankErrorBase
except Exception:  # pragma: no cover - runtime package is optional across branches.
    _BankErrorBase = RuntimeError


class BankAuthError(_BankErrorBase):
    """Raised when an encrypted bank cannot be authenticated or decrypted."""


def _require_fernet() -> tuple[Any, Any]:
    try:
        from cryptography.fernet import Fernet, InvalidToken
    except ImportError as exc:
        raise BankAuthError(
            "Encrypted bank support requires the optional security extra: "
            "pip install 'deltamemory[security]' or pip install cryptography>=42"
        ) from exc
    return Fernet, InvalidToken


def _serialize_bank(bank: dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(bank, buf)
    return buf.getvalue()


def save_encrypted(bank: dict[str, torch.Tensor], path: str, key: bytes) -> None:
    """Encrypt and save a tensor bank with Fernet without logging the key."""
    Fernet, _InvalidToken = _require_fernet()
    try:
        plaintext = _serialize_bank(bank)
        token = Fernet(key).encrypt(plaintext)
    except Exception as exc:
        raise BankAuthError("failed to encrypt bank; verify the Fernet key") from exc

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(token)
    audit_event(event_type="bank_store", vector_hash=bytes_sha256(plaintext))


def load_encrypted(path: str, key: bytes) -> dict[str, torch.Tensor]:
    """Decrypt and load a tensor bank using ``torch.load(..., weights_only=True)``."""
    Fernet, InvalidToken = _require_fernet()
    token = Path(path).read_bytes()
    try:
        plaintext = Fernet(key).decrypt(token)
    except (InvalidToken, ValueError, TypeError) as exc:
        raise BankAuthError("failed to authenticate encrypted bank") from exc

    try:
        bank = torch.load(io.BytesIO(plaintext), weights_only=True)
    except Exception as exc:
        raise BankAuthError("failed to load decrypted bank weights") from exc
    if not isinstance(bank, dict) or not all(isinstance(v, torch.Tensor) for v in bank.values()):
        raise BankAuthError("decrypted payload is not a dict[str, torch.Tensor]")
    audit_event(event_type="bank_load", vector_hash=bytes_sha256(plaintext))
    return bank


__all__ = ["BankAuthError", "load_encrypted", "save_encrypted"]
