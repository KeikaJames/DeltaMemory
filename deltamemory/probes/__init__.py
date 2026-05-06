"""Memory↔attention probes (decoupled from the AttnNativeBank hot path)."""

from .extra_probes import (
    bank_eldest_score_margin,
    top_k_softmax_entropy,
)

__all__ = ["top_k_softmax_entropy", "bank_eldest_score_margin"]
