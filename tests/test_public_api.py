"""A5a: assert every name in ``deltamemory.__all__`` is importable.

Conditionally-loaded names (e.g. ``MoeAttnNativePatcher``) are skipped if
the underlying module isn't present (b-w5-moe in flight).
"""
from __future__ import annotations

import pytest

import deltamemory


_CONDITIONAL = {"MoeAttnNativePatcher"}


@pytest.mark.parametrize("name", deltamemory.__all__)
def test_public_api_imports(name: str) -> None:
    if name in _CONDITIONAL and not getattr(deltamemory, "_has_moe", False):
        pytest.skip(f"{name} is conditionally loaded; module not present")
    assert hasattr(deltamemory, name), f"{name} missing from deltamemory"
    obj = getattr(deltamemory, name)
    assert obj is not None, f"deltamemory.{name} is None"
