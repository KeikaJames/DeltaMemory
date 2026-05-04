"""Delta Memory — frozen-LLM external memory via attention-side bank injection.

Public API (Phase R/S, recommended for new code)
------------------------------------------------
* :class:`AttnNativePatcher`, :class:`AttnNativeBank` — the canonical bank
  implementation used by Phase Q+ experiments.
* :func:`pick_adapter` — auto-pick the right
  :class:`~deltamemory.memory.arch_adapter.ArchAdapter` for a HF model.
* :class:`LOPIConfig`, :class:`LOPIState`, :func:`apply_lopi` — Dynamic LOPI
  injector (Phase R).  ``profile_mode='auto'`` activates U-LOPI (Phase S).
* :func:`profile_residuals`, :class:`LOPIProfile` — U-LOPI auto-calibration.
* :func:`save_bank`, :func:`load_bank` — disk persistence for the bank,
  including the ``LOPIProfile`` (Phase S).

Legacy entry points (``deltamemory.gemma``, ``deltamemory.engine``,
``deltamemory.memory.attention_store.AttentionMemoryStore``) remain importable
but are not part of the recommended path.
"""
from __future__ import annotations

__version__ = "0.3.5"  # Phase S — U-LOPI

# Lazy-friendly: top-level imports must not break if optional deps (faiss)
# are missing on a given machine; persistence/profiler tolerate this.
from deltamemory.memory.attn_native_bank import (
    AttnNativeBank,
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.arch_adapter import pick_adapter
from deltamemory.memory.lopi import (
    LOPIConfig,
    LOPIState,
    apply_lopi,
    derivative_gate,
    layer_gaussian_weight,
    orthogonal_novelty,
)
from deltamemory.memory.lopi_profiler import (
    LOPIProfile,
    default_profile_corpus,
    load_profile,
    profile_residuals,
    save_profile,
)
from deltamemory.memory.bank_persistence import (
    compute_config_sha,
    list_banks,
    load_bank,
    resolve_location,
    save_bank,
)

__all__ = [
    "__version__",
    # bank
    "AttnNativeBank", "AttnNativePatcher", "fresh_bank", "write_fact",
    # arch
    "pick_adapter",
    # lopi
    "LOPIConfig", "LOPIState", "apply_lopi",
    "derivative_gate", "layer_gaussian_weight", "orthogonal_novelty",
    # profiler
    "LOPIProfile", "default_profile_corpus", "load_profile",
    "profile_residuals", "save_profile",
    # persistence
    "compute_config_sha", "list_banks", "load_bank",
    "resolve_location", "save_bank",
]
