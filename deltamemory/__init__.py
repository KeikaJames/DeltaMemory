"""Mneme — frozen-LLM external memory via attention-side bank injection.

Public API for v0.4. See ``docs/`` for architecture notes
(``docs/theory/``), per-phase reports under ``experiments/W*/``, and
the v0.4 ship status in ``CHANGELOG.md``. The recommended entry points
are :class:`AttnNativePatcher` + :class:`AttnNativeBank` for the bank,
:func:`pick_adapter` for HF-model auto-detection, and
:class:`CAAInjector` / :class:`DiagnosticRecorder` for the W.4 baseline
and instrumentation paths.

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
* :class:`CAAInjector` — W.4 contrastive activation-addition baseline.
* :class:`DiagnosticRecorder` — Phase X.1 instrumentation.
* :class:`MoeAttnNativePatcher` — MoE-aware patcher (Phase W.5, optional;
  ``None`` if the module is not yet present).

Legacy entry points (``deltamemory.gemma``, ``deltamemory.engine``,
``deltamemory.memory.attention_store.AttentionMemoryStore``) remain importable
but are not part of the recommended path.
"""
from __future__ import annotations

__version__ = "0.4.0"  # Phase X.1 — DiagnosticRecorder

# Lazy-friendly: top-level imports must not break if optional deps (faiss)
# are missing on a given machine; persistence/profiler tolerate this.
from deltamemory.memory.arch_adapter import pick_adapter
from deltamemory.memory.attn_native_bank import (
    AttnNativeBank,
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.bank_persistence import (
    compute_config_sha,
    list_banks,
    load_bank,
    resolve_location,
    save_bank,
)
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
from deltamemory.diagnostics import DiagnosticRecorder
from deltamemory.memory.caa_injector import CAAInjector

try:
    from deltamemory.memory.moe_attn_patcher import MoeAttnNativePatcher
    _has_moe = True
except ImportError:
    MoeAttnNativePatcher = None
    _has_moe = False

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
    # diagnostics (Phase X.1)
    "DiagnosticRecorder",
    # W.4 baseline
    "CAAInjector",
    # W.5 MoE (conditional; None if module missing)
    "MoeAttnNativePatcher",
]
