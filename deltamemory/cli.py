"""Deprecated shim. Use :mod:`deltamemory.bench` (``dm-bench``) instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "deltamemory.cli is deprecated; use 'dm-bench' "
    "(python -m deltamemory.bench) instead.",
    DeprecationWarning,
    stacklevel=2,
)

from deltamemory.bench.__main__ import main as _main


def main() -> int:
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
