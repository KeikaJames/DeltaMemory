"""deltamemory.injection — injection-side utilities for AttnNativeBank."""
from deltamemory.injection.safe_alpha import (
    SafeAlphaScheduler,
    compute_safe_alpha_threshold,
    validate_scheduler_vs_naive,
    CLIFF_LO_DEFAULT,
    CLIFF_HI_DEFAULT,
    CLIFF_THRESHOLD_DEFAULT,
    POST_CLIFF_ALPHA_DEFAULT,
)

__all__ = [
    "SafeAlphaScheduler",
    "compute_safe_alpha_threshold",
    "validate_scheduler_vs_naive",
    "CLIFF_LO_DEFAULT",
    "CLIFF_HI_DEFAULT",
    "CLIFF_THRESHOLD_DEFAULT",
    "POST_CLIFF_ALPHA_DEFAULT",
]
