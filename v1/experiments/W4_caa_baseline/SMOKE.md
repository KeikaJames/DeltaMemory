# W.4 CAA Baseline - Smoke Run

**Status**: PASS (alpha=0 bit-equality verified)
**Date**: 2026-05-04
**Branch**: `feat/v04-w4-caa-runner`
**Base commit**: `ad579c4d`
**Device / dtype**: MPS / bfloat16

## 1. Configuration

Smoke is a strict subset of the PREREG section 2 grid:

| Axis | Smoke value | Full grid |
|---|---|---|
| Model | `gpt2-medium` | 5 models |
| Method | `none`, `caa`, `lopi_default` | 3 methods |
| alpha | 0.0, 1.0 | 7 levels |
| Seed | 0 | 3 seeds |
| Prompts | gold_001 ... gold_005 | 30 prompts |

Total emitted rows: **21** (20 real cells + 1 `method_unsupported` sentinel for
`gpt2-medium x lopi_default`, since GPT-2 has no `AttnNativePatcher` path; this
matches the W.2 fallback policy).

## 2. alpha=0 bit-equality (red-line)

Every alpha=0 cell must satisfy `|drift| < 1e-4`. Observed:

| method | n | max\|drift\| |
|---|---|---|
| `none` | 5 | 0.000e+00 |
| `caa`  | 5 | 0.000e+00 |

**Verdict: PASS** (0 redline violations; max-abs-diff = 0.000e+00 nats).

The CAA injector's documented short-circuit (`if alpha == 0.0: return output`)
preserves the residual stream untouched, and the `none` arm on GPT-2 runs a pure
baseline forward, so both arms recover the baseline logits exactly.

## 3. alpha=1.0 sanity check

CAA injection must produce a non-trivial residual perturbation at alpha=1.0:

| prompt | drift (nats) |
|---|---|
| gold_001 | +0.4520 |
| gold_002 | +0.3249 |
| gold_003 | +0.2795 |
| gold_004 | +0.3847 |
| gold_005 | +0.3305 |

Mean drift: **+0.354 nats** at alpha=1.0 on GPT-2-medium with 16-pair
contrastive calibration; steering vector L2 norm = 65.26 (resolved injection
layer = 8 of 24 via LOPI profiler `mu_arch`).

Drift is positive, indicating CAA at unit alpha *increases* per-token NLL on
the neutral gold set. This is the expected behaviour of unconstrained additive
residual injection at default scale and matches the X.3 unit-test sanity bound.
The full grid will measure whether the paired contrast `caa - none` is
significantly negative anywhere in the alpha sweep.

## 4. Artifacts

- `cells_smoke.jsonl` (21 rows, append-only, schema-locked)
- `env.json` (torch 2.10.0, transformers 5.2.0, commit ad579c4d, MPS, bfloat16)

## 5. Runtime

20 forward passes in **~12 seconds** wall-clock on the Apple M-series MPS
backend, ~0.6 s / forward. Extrapolated to the full grid (9450 cells): roughly
1.6 hours of pure forward time on GPT-2-medium scale, with the larger models
(Qwen2.5-1.5B, gemma-3-1b-it) dominating actual wall-clock. Full-grid execution
is deferred until the 128 GB hardware is available per the user's scheduling
constraint.

## 6. Tests

- `tests/test_w4_smoke.py` - 2/2 PASS
  - `test_w4_run_module_imports_without_side_effects`
  - `test_w4_smoke_output_schema`
- `pytest tests/ --ignore=tests/conservation_real_models.py --ignore=tests/test_moe_adapter.py` - 140 passed, 6 skipped, 0 failed

## 7. Pre-flight clearance

All six PREREG section 4 pre-conditions hold:

1. CAA module exists and its 7 unit tests pass (X.3, prior commit).
2. alpha=0 bit-equality reproduced here.
3. LAMA T-REx (500) and ConceptNet (500) datasets locked at commit 5d044870.
4. PREREG.md committed before any cells.jsonl row was written.
5. Five PREREG models declared in `MODELS`; substitution policy implemented
   (load failure produces `model_substituted=true` rows, run continues).
6. Three arms `none`, `lopi_default`, `caa` all wired; LOPI uses the shipped
   defaults (orthogonal=True, gaussian=True, derivative=True), not the W-T3
   fixes, per W.3 ruling.

The runner is cleared for the full sweep when 128 GB is available.
