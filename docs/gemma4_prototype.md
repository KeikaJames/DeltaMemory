# Gemma4 RCV-HC Prototype Runbook

This is the practical cleanroom path for RCV-HC:

```text
long context
-> frozen Gemma4 forward
-> per-layer Raw/Delta attention memory
-> external store
-> top-k retrieval
-> Q/K/V intervention
-> answer metrics and attention-memory trace
```

It is not a RAG prompt-insertion path. Retrieved source snippets are written to
the trace for debugging only and are not appended to the question prompt.

## Mock Wiring Run

Use this for fast local validation:

```bash
python scripts/run_gemma4_prototype.py \
  --model mock-gemma \
  --device cpu \
  --dtype float32 \
  --block-size 32 \
  --memory-dim 128
```

Expected output:

- `reports/cleanroom/gemma4_prototype_report.md`
- `reports/cleanroom/gemma4_prototype_summary.json`

## Real Gemma4 Run

Use this when Hugging Face access and local memory are ready:

```bash
python scripts/run_gemma4_prototype.py \
  --model google/gemma-4-E2B \
  --device auto \
  --dtype bfloat16 \
  --block-size 128 \
  --memory-dim 512 \
  --top-k 4 \
  --alpha-scale 0.2 \
  --gate-bias -1
```

The base model remains frozen. The report records `trainable_base_params`, which
must be `0`.

## Interpreting Results

Engineering success:

- ingest creates memory blocks.
- ask runs all modes.
- `delta_qv` has non-zero `q_delta_norm` and `v_delta_norm`.
- `delta_qv_zero` is near no-memory.
- `delta_qv_force_gate` changes Q/V norms.

Scientific signal:

- `delta_qv` must beat zero and random controls.
- block-aligned `delta_qv` must beat shuffled.

If those controls do not pass, the run is prototype wiring only, not evidence of
effectiveness.

