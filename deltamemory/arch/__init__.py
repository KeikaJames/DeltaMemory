"""DeltaMemory architecture adapters.

This package provides architecture-specific adapters that expose a uniform
interface for querying MoE router gates and expert pools.  The adapters are
used by the mHC per-expert shield (Phase X.2 / W.5) to obtain per-token
expert routing information.

Available adapters
------------------
- :class:`MoEArchAdapter`  — abstract base class
- :class:`Qwen3MoEAdapter` — Qwen/Qwen3-30B-A3B and Qwen3-MoE-A1.5B family
- :class:`MixtralAdapter`  — mistralai/Mixtral-8x7B-v0.1

Architectural note
------------------
All currently supported MoE models (Mixtral, Qwen3-MoE, DeepSeek-V3) use
**FFN-MoE with dense attention**.  MoE routing operates on the FFN sublayer;
the attention sublayer is shared across all experts and therefore the
bank-column cap from ``mhc_shield`` operates on a single merged attention
weight matrix that is not natively factored per expert.

The per-expert cap formula in ``mhc_shield.apply_shield_per_expert`` uses
the FFN router gates as proxy weights to bucket attention queries into
"per-expert populations" and applies an independent column cap within each
bucket.  This is a conservative approximation that bounds each expert's
effective bank injection energy; it does **not** correspond to a true
MoE-Attention architecture where V projections are per-expert.

True MoE-Attention (expert-specific W_v) would allow an exact per-expert
cap without any approximation.  As of Phase X.2, no such architecture is
supported by this adapter, and the per-expert mode is opt-in infrastructure
only — default shield behaviour (global cap) is unchanged.

W.5 re-visit note: before running the MoE sweep, confirm whether the
router gates from the FFN sublayer at layer L are the correct proxies for
the attention cap at layer L or layer L-1.
"""

from deltamemory.arch.moe_adapter import (
    MixtralAdapter,
    MoEArchAdapter,
    Qwen3MoEAdapter,
)

__all__ = [
    "MoEArchAdapter",
    "Qwen3MoEAdapter",
    "MixtralAdapter",
]
