"""Deprecated path; module relocated to deltamemory.legacy.engine."""
import warnings as _warnings
_warnings.warn(
    "deltamemory.engine has moved to deltamemory.legacy.engine; "
    "please update your imports. This shim will be removed in v0.6.",
    DeprecationWarning, stacklevel=2,
)
from deltamemory.legacy.engine import *  # noqa: F401,F403
from deltamemory.legacy import engine as _impl  # noqa: F401
import sys as _sys
_sys.modules[__name__] = _impl
