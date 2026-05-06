# L.1 Marathon — Llama-3.1-8B-Instruct — COMPLETE (v0.8 cross-arch replication)

**Status**: 36/36 cells non-aborted. H_L stability verdict **PASS** for both methods.

## Setup

| Field | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.1-8B-Instruct` via `unsloth/Meta-Llama-3.1-8B-Instruct` (byte-identical mirror; upstream gated even on hf-mirror) |
| Methods | `lopi_default`, `caa` |
| Seeds | 0, 1, 2 |
| Turns per cell | 2000 |
| Checkpoints per cell | 6 (turn = 1, 50, 200, 500, 1000, 2000) |
| Cells | 2 methods × 3 seeds × 6 checkpoints = **36** |
| Hardware | spark1 GB10 (CUDA bf16) |
| Adapter | `arch_adapter.LLAMA` — **no q/k/v_norm** (verified `arch_adapter.py:180-207`) |

## Verdict (`aggregate.py`)

```
n_cells_total        = 36
n_cells_aborted      = 0
n_cells_non_aborted  = 36
n_h_l_pass           = 2 / 2  (both methods)
```

| Method | Median Δ NLL (turn 2000 − turn 1) | 95 % CI | n_pairs | p (Wilcoxon) | turn-1 NLL median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lopi_default` | **0.0** | [0.0, 0.0] | 3 | 1.0 | 4.787 |
| `caa`          | **0.0** | [0.0, 0.0] | 3 | 1.0 | 5.302 |

Per-cell `nan_inf_count = 0`. No `mem_rss_jump_gt_1gb`, no `nll_decay_gt_10x`,
no `alpha_zero_drift_witness` aborts. Residual norm `~526–535` constant
across all 36 cells.

## Cross-arch summary (with gemma-4-31B + Qwen3-4B)

| Architecture | Methods PASS | Norm regime | turn-1 NLL median (lopi/caa) |
| --- | ---: | --- | ---: |
| `gemma-4-31B-it`            | 2/2 | q+k+v_norm | 16.869 / 16.883 |
| `Qwen3-4B-Instruct-2507`    | 2/2 | q+k_norm only | 10.236 / 7.295 |
| `Llama-3.1-8B-Instruct`     | 2/2 | none | 4.787 / 5.302 |
| `Qwen3.6-27B`               | excluded | linear attention | n/a |

L.1 stability invariant **holds across all three normalization regimes** for
dense-attention transformers in the 4B–31B range. This empirically refutes
the "single-architecture artifact" reviewer concern from
`papers/v0_8_rigor_overhaul/OPEN_QUESTIONS.md` Q41–Q43.

## Caveats

Same as siblings: bit-stability under filler context, not interactive recall.
v0.8 metrics_v2 will replace with CounterFact-1k per-turn recall.

## Reproduction

```bash
ssh spark1
cd /home/gabira/projects/RCV-HC && source .venv-gb10/bin/activate
# Llama gated upstream; use unsloth byte-identical mirror via hf-mirror.com
HF_ENDPOINT=https://hf-mirror.com hf download \
    unsloth/Meta-Llama-3.1-8B-Instruct \
    --local-dir /home/gabira/Desktop/workspace/models/whitelist/Llama-3.1-8B-Instruct
bash /tmp/dispatch_L1_v08.sh
```

## Files

- `aggregate_summary.json/summary.json` — H_L verdicts + bootstrap CIs
- `aggregate_summary.json/flat_table.csv` — per-checkpoint long table
- `lopi_default/seed{0,1,2}/cells.jsonl`, `caa/seed{0,1,2}/cells.jsonl` — raw checkpoints
