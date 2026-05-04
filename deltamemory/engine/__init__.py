"""Cleanroom Delta Memory engine.

Deprecated since v0.3 (Phase Q): use ``deltamemory.AttnNativePatcher`` /
``fresh_bank`` / ``LOPIConfig`` (Phase R+) instead.
"""

import warnings as _w
_w.warn(
    "deltamemory.engine is the pre-Phase-Q legacy path; new code should use "
    "deltamemory.AttnNativePatcher / fresh_bank / LOPIConfig (Phase R+).",
    DeprecationWarning, stacklevel=2,
)
