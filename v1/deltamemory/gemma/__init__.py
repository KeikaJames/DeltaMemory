"""Deprecated path; module relocated to deltamemory.legacy.gemma."""
import warnings as _warnings
_warnings.warn(
    "deltamemory.gemma has moved to deltamemory.legacy.gemma; "
    "please update your imports. This shim will be removed in v0.6.",
    DeprecationWarning, stacklevel=2,
)
from deltamemory.legacy.gemma import *  # noqa: F401,F403
from deltamemory.legacy import gemma as _impl  # noqa: F401
import sys as _sys
_sys.modules[__name__] = _impl
