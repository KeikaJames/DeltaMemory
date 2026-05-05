"""Fernet-encrypted persistence helpers for tensor-only Mneme banks."""
from __future__ import annotations

import io
import os
import tempfile
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
    """Encrypt and save a tensor bank with Fernet without logging the key.
    
    Performs atomic write using tempfile + os.replace to ensure no partial
    files remain on interruption.
    """
    Fernet, _InvalidToken = _require_fernet()
    try:
        plaintext = _serialize_bank(bank)
        token = Fernet(key).encrypt(plaintext)
    except Exception as exc:
        raise BankAuthError("failed to encrypt bank; verify the Fernet key") from exc

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=Path(path).parent,
            delete=False,
            suffix=".tmp"
        ) as tmp:
            tmp_file = tmp.name
            tmp.write(token)
        os.replace(tmp_file, path)
    except Exception:
        if tmp_file and Path(tmp_file).exists():
            os.remove(tmp_file)
        raise
    
    audit_event(event_type="bank_store", vector_hash=bytes_sha256(plaintext))


def load_encrypted(path: str, key: bytes, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Decrypt and load a tensor bank using ``torch.load(..., weights_only=True)``.
    
    Args:
        path: Path to the encrypted bank file.
        key: Fernet encryption key.
        device: Device to map tensors to (default: "cpu"). Pass "cuda" or other
                device strings to override. This ensures banks saved from GPU/meta
                writers can be restored on CPU-only readers.
    
    Returns:
        Decrypted dict[str, torch.Tensor] with all tensors on the specified device.
    """
    Fernet, InvalidToken = _require_fernet()
    token = Path(path).read_bytes()
    try:
        plaintext = Fernet(key).decrypt(token)
    except (InvalidToken, ValueError, TypeError) as exc:
        raise BankAuthError("failed to authenticate encrypted bank") from exc

    try:
        bank = torch.load(io.BytesIO(plaintext), weights_only=True, map_location=device)
    except Exception as exc:
        raise BankAuthError("failed to load decrypted bank weights") from exc
    if not isinstance(bank, dict) or not all(isinstance(v, torch.Tensor) for v in bank.values()):
        raise BankAuthError("decrypted payload is not a dict[str, torch.Tensor]")
    audit_event(event_type="bank_load", vector_hash=bytes_sha256(plaintext))
    return bank


__all__ = ["BankAuthError", "load_encrypted", "save_encrypted"]
