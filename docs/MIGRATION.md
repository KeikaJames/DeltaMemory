# DeltaMemory — Migration Guide (legacy → Phase R/S canonical APIs)

This document maps the **pre-Phase-Q legacy modules** (now emitting
`DeprecationWarning` at import) to their **Phase R / Phase S canonical**
replacements. The legacy code is still importable and exercised by the
test-suite, but new work should target the canonical paths.

> See also: [`scripts/legacy/README.md`](../scripts/legacy/README.md) for the
> archived stage runners (Phase ≤ 11) that were moved out of `scripts/` in
> commit `ecb66288`.

---

## 1. Module-level mapping

| Legacy import (deprecated)                                       | Canonical replacement (Phase R/S)                                      |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `from deltamemory.gemma import …`                                | `from deltamemory import AttnNativePatcher` + `pick_adapter` + Gemma adapter |
| `deltamemory.gemma.attention_injector.GemmaWithMemory`           | `deltamemory.AttnNativePatcher(model, adapter=pick_adapter(model))`    |
| `deltamemory.gemma.fast_weight_injector`                         | Folded into `AttnNativePatcher` (per-layer K/V concat in attention).   |
| `deltamemory.gemma.model_adapter`                                | `deltamemory.memory.arch_adapter.pick_adapter(model)`                  |
| `from deltamemory.engine import …`                               | `from deltamemory import AttnNativeBank, fresh_bank, write_fact`       |
| `deltamemory.engine.attention_memory_engine.AttentionMemoryEngine` | `AttnNativePatcher` + `AttnNativeBank` (read/write split)            |
| `deltamemory.engine.delta_dataset` / `delta_training`            | No canonical replacement; superseded by Phase R-5 paired-arms harness. |
| `deltamemory.engine.prototype` (Gemma-4 prototype)               | `examples/` quickstarts on top of `AttnNativePatcher`.                 |
| `deltamemory.memory.attention_store.AttentionMemoryStore`        | `deltamemory.AttnNativeBank` (per-layer K/V semantic; carries reasons / address keys). |

The legacy modules now emit a single `DeprecationWarning` on first import (see
commit `b15969e5`). They are **not** removed — downstream callers and the
remaining legacy tests continue to work.

---

## 2. Phase R LOPI knobs — and what changed in Phase S

Phase R introduced **LOPI** (Layer-Of-Peak-Influence) gating in
`deltamemory.memory.lopi.LOPIConfig`. The relevant knobs are:

| Knob                       | Role                                                                 |
| -------------------------- | -------------------------------------------------------------------- |
| `mu_arch`                  | Architectural target layer (peak-influence layer for the backbone).  |
| `sigma_arch`               | Width of the Gaussian gate around `mu_arch`.                         |
| `eta_*` (drift coefs)      | Per-arm online drift terms.                                          |
| `floor` / `ceiling`        | Hard clamps on the per-layer gate.                                   |
| `profile_mode`             | **NEW in Phase S:** how `(mu_arch, sigma_arch)` is obtained.         |

### `profile_mode='static'` (Phase R behaviour)

`mu_arch` and `sigma_arch` come from the hardcoded per-arch defaults shipped in
`LOPIConfig`. These were tuned offline on Gemma-2 / Phi-3 / Llama-3 in the
Phase Q–R cleanroom runs.

### `profile_mode='auto'` (Phase S — U-LOPI)

`AttnNativePatcher` runs a **one-shot residual-stream profiler**
(`deltamemory.memory.lopi_profiler.profile_residuals`) over a small calibration
corpus the first time the bank is opened against a model. The profiler
returns a `LOPIProfile(mu_base, sigma_base)` of length `model.num_layers`,
and the `LOPIConfig`'s `mu_arch` is set to `argmax_l sigma_base[l]` (with a
low-index tiebreak for bit-stability across save/load), and `sigma_arch` is
derived from the coefficient of variation of `sigma_base` (heterogeneous
profiles get `eta_sigma=0.7`; homogeneous ones a tighter default).

Profiles are persisted alongside the bank and reloaded bit-stably on the next
open. See `tests/test_lopi_profiler.py` and `tests/test_lopi_universal.py` for
the contract.

**Migration note:** `profile_mode='auto'` is the new default for fresh banks.
Banks created on Phase R (`profile_mode='static'`) continue to load — see §3.

---

## 3. Bank schema bumps: `lopi_v33` → `ulopi_v35` → `ulopi_v36`

`deltamemory.memory.bank_persistence.VERSION` advanced to **`ulopi_v35`** in
Phase S and **`ulopi_v36`** in R-7. The previous on-disk formats `lopi_v33`
and `ulopi_v35` remain loadable via the `_LEGACY_VERSIONS` tuple in
`bank_persistence.py`.

### What changed in `ulopi_v35`

- The `LOPIConfig` block now includes `profile_mode` plus the Phase-S derived
  fields (`mu_arch`, `sigma_arch` are written as the *resolved* values, not
  the static defaults).
- A `LOPIProfile` sidecar (mu_base / sigma_base / fingerprints) is stored
  alongside the safetensors bank when `profile_mode='auto'`.

### What changed in `ulopi_v36`

- `AttnNativeBank` persists `value_scale_mode`, `value_target_rms`, and
  `value_scale_eps`.
- The default `value_scale_mode='auto_rms_cap'` leaves Gemma-style native
  `v_norm` layers untouched and caps no-v_norm family `M_V` values at fixed
  per-head RMS without amplifying already-small V activations. This makes
  `alpha` less architecture-scale-dependent while preserving `alpha=0`
  bit-equality.

### Migrating a persisted Phase-R bank

If you have `lopi_v33` banks on disk:

1. **Read-only continues to work** — `load_bank(path)` accepts `lopi_v33` and
   surfaces it through the same `AttnNativeBank` API.
2. **To upgrade in place to the current schema (`ulopi_v36`):**

   ```python
   from deltamemory.memory.bank_persistence import load_bank, save_bank
   from deltamemory.memory.lopi import LOPIConfig

   bank = load_bank(old_path, model=model)        # accepts lopi_v33
   bank.lopi.profile_mode = "auto"                # opt into U-LOPI
   # First read against `model` will materialise the profile.
   save_bank(bank, new_path)                      # writes ulopi_v36
   ```

   The migration is one-way: `save_bank` always writes the current `VERSION`.
   If you need to keep a `lopi_v33` copy, copy the file aside before
   round-tripping.

3. **Cross-arch banks** (e.g., trained on Gemma-2, replayed on Phi-3) will
   *re-profile* on first read of the new arch, so plan for one extra forward
   pass on the calibration corpus.

---

## 4. Test-suite handling of the deprecation warnings

Tests that **explicitly** exercise legacy modules (e.g.
`tests/test_clean_attention_store.py`, `tests/test_clean_engine_smoke.py`,
`tests/test_delta_experiment.py`, etc.) install an import-time
`warnings.filterwarnings("ignore", category=DeprecationWarning)` plus a
`pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")` so
they pass under `pytest -W error::DeprecationWarning`.

New tests should not import from `deltamemory.gemma`,
`deltamemory.engine`, or `deltamemory.memory.attention_store`. Use
`deltamemory.AttnNativePatcher`, `AttnNativeBank`, `fresh_bank`,
`write_fact`, `LOPIConfig`, and `pick_adapter` instead.

---

## 5. Archived runners

The pre-Phase-Q stage runners (`run_clean_demo`, `run_stage11_*`,
`run_stage12_multimodel`, `run_stage13b_robust`, …) live under
[`scripts/legacy/`](../scripts/legacy/) since commit `ecb66288`. They are
preserved verbatim for reproducibility of historical reports under
`reports/` and are **not** part of the current Phase-S regression surface.
