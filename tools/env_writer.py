"""Helper to emit env.json files conforming to docs/authenticity.md.

Usage from a runner::

    from tools.env_writer import write_env_json

    write_env_json(
        out_dir=Path("experiments/A_ablation"),
        prereg_version="A.v1",
        dataset_sha1=sha1_of("experiments/datasets/counterfact_60.jsonl"),
        device="mps",
        dtype="bfloat16",
        cli_argv=sys.argv,
    )

Keeps every runner from hand-rolling its own dict and forgetting fields.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha1_of(path: str | Path) -> str:
    """sha1 of a file's bytes."""
    h = hashlib.sha1()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str, default: str = "") -> str:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
        return out or default
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


def _torch_version() -> str:
    try:
        import torch  # type: ignore

        return torch.__version__
    except Exception:
        return "n/a"


def _transformers_version() -> str:
    try:
        import transformers  # type: ignore

        return transformers.__version__
    except Exception:
        return "n/a"


def build_env(
    *,
    prereg_version: str,
    dataset_sha1: str | dict[str, str],
    device: str,
    dtype: str,
    cli_argv: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD", default="0" * 40)
    porcelain = _git("status", "--porcelain")
    dirty = bool(porcelain)
    if dirty:
        diff = _git("diff")
        dirty_diff_sha1 = hashlib.sha1(diff.encode()).hexdigest()
    else:
        dirty_diff_sha1 = "0" * 40

    env: dict[str, Any] = {
        "commit": commit,
        "dirty": dirty,
        "dirty_diff_sha1": dirty_diff_sha1,
        "prereg_version": prereg_version,
        "dataset_sha1": dataset_sha1,
        "torch": _torch_version(),
        "transformers": _transformers_version(),
        "python": platform.python_version(),
        "device": device,
        "dtype": dtype,
        "started_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
    }
    if cli_argv is not None:
        env["cli_argv"] = list(cli_argv)
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
        env["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception:
        pass
    try:
        import numpy  # type: ignore

        env["numpy"] = numpy.__version__
    except Exception:
        pass
    if extra:
        env.update(extra)
    return env


def write_env_json(
    out_dir: str | Path,
    *,
    prereg_version: str,
    dataset_sha1: str | dict[str, str],
    device: str,
    dtype: str,
    cli_argv: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write env.json into ``out_dir`` and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cli_argv is None:
        cli_argv = sys.argv
    env = build_env(
        prereg_version=prereg_version,
        dataset_sha1=dataset_sha1,
        device=device,
        dtype=dtype,
        cli_argv=cli_argv,
        extra=extra,
    )
    path = out_dir / "env.json"
    path.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    return path


__all__ = ["build_env", "write_env_json", "sha1_of"]
