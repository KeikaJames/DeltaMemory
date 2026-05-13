# Test Pyramid

Snapshot of the `tests/` suite, organised by tier. Goal: every public surface
has at least one unit-level guard, every cross-module contract has an
integration test, and the bit-equality / authenticity invariants are checked
end-to-end.

Total: **441 tests** collected (CI green on push, see `.github/workflows/ci.yml`).

## L1 — Unit (≈70%)

Pure-function / single-module tests. Fast (< 0.5 s each), no model load.

| Area | Files |
| --- | --- |
| Memory bank lifecycle | `test_attn_native_bank.py`, `test_attn_native_bank_validation.py`, `test_x7_bank_lifecycle.py` |
| Bank persistence | `test_bank_persistence.py` (incl. heterogeneous KV-heads, decay-aware metadata) |
| Bank decay / tiering / compression | `test_bank_decay.py`, `test_bank_tiering.py`, `test_bank_compression.py`, `test_bank_importance.py` |
| LOPI (orthogonal projection) | `test_lopi_module.py`, `test_lopi_profiler.py` |
| Injectors (CAA / SCAR / ROME / kproj) | `test_caa_injector.py`, `test_scar_injector.py`, `test_rome_writer.py`, `test_k_projector.py` |
| Arch adapters | `test_arch_adapter_coverage.py`, `test_clean_gemma_injector_import_safe.py` |
| Capture policy | `test_capture_policy.py` |
| Diagnostics & schemas | `test_config_schemas.py`, `test_diagnostics*.py` |

## L2 — Integration (≈25%)

Multiple modules wired together; may load a tiny dummy model.

| Area | Files |
| --- | --- |
| Attention + bank merged-softmax bit-equality | `test_a_ablation.py`, `test_a2_causal_mask.py`, `test_attn_native_bank.py::test_empty_bank_is_bit_equal` |
| Patcher + adapter + bank end-to-end | `test_injector_contract.py`, `test_clean_engine_smoke.py` |
| Delta-experiment runner | `test_delta_experiment.py`, `test_delta_training_prototype.py`, `test_gemma4_prototype_runner.py` |
| Bench CLI | `test_bench_cli.py`, `test_bench_smoke.py` (marked `slow`) |
| Experiment aggregation | `test_experiment_aggregates.py`, `test_aggregate_flagship_q2.py`, `test_l2_figures.py`, `test_w6_smoke.py`, `test_w7_smoke.py` |

## L3 — Authenticity / E2E invariants (≈5%)

Repository-wide guards executed every CI run.

| Invariant | Test |
| --- | --- |
| Every committed `runs/<dir>/` has `cells.jsonl` + `env.json` (or is exempt with documented reason) | `test_run_authenticity.py` |
| Authenticity tooling matches schema | `test_check_authenticity.py` |
| Branch / PR helper script | `test_branch_pr_helper.sh` |
| Full-suite smoke | `test_full_suite_still_passes.sh` |

## Known gaps (open debt)

- **No CUDA-only path**: bit-equality tests run on CPU (Mac sandbox); the
  spark1 GB10 path is verified manually via `D1_bit_equality_v1` (run is
  pending — currently exempt from L3 guard).
- **No vLLM integration test**: `integrations/vllm/bank_attached_llm.py` is
  exercised manually only; `P1` debt item tracks adding an e2e harness once
  vLLM ≥ 0.4 is on the runner.
- **Long-context (W.7/E)**: only smoke-grid in CI; full sweeps need GPU.
- **Cross-arch matrix**: Llama-4-Scout / gpt-oss-120b / DeepSeek-V4 still
  run via dispatch scripts on spark1, no CI mirror.

## Running locally

```bash
pytest -q                       # full suite, ~2 min on CPU
pytest -q -m "not slow"         # skip slow benchmarks
pytest -q tests/test_attn_native_bank.py  # one module
```

CI runs on Python 3.11 / 3.12 / 3.13 plus a `ruff check .` job — see
`.github/workflows/ci.yml`.
