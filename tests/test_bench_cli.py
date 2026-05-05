"""Tests for ``dm-bench`` CLI router (Phase G-1)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _has_hf_cache(model_id: str) -> bool:
    """Return True if ``model_id`` looks present in the local HF hub cache."""
    base = Path(
        os.environ.get("HF_HOME")
        or os.path.expanduser("~/.cache/huggingface")
    )
    safe = "models--" + model_id.replace("/", "--")
    return (base / "hub" / safe).exists() or (base / safe).exists()


def _run(args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "deltamemory.bench", *args],
        capture_output=True,
        text=True,
        **kw,
    )


def test_info_subcommand():
    proc = _run(["info"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "AttnNativePatcher" in out
    assert "torch" in out
    assert "version" in out


def test_help_subcommand():
    proc = _run(["--help"])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for sub in ("info", "smoke", "profile", "replay"):
        assert sub in out, f"subcommand {sub!r} missing from help"


# Pick the smallest cached supported model.  AttnNativePatcher only handles
# Gemma / Qwen / Llama families, so plain GPT-2 / tiny-gpt2 won't work.
_SMOKE_MODEL = "Qwen/Qwen2.5-0.5B"


@pytest.mark.skipif(
    not _has_hf_cache(_SMOKE_MODEL),
    reason=f"{_SMOKE_MODEL} not in local HF cache",
)
def test_smoke_alpha_zero_bit_equal():
    proc = _run(
        ["smoke", "--model", _SMOKE_MODEL, "--device", "cpu", "--alpha", "0"],
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert "redline OK" in proc.stdout, proc.stdout
