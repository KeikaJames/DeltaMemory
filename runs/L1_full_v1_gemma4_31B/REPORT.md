# L.1 Marathon — gemma-4-31B-it — COMPLETE

**Status**: 36/36 cells non-aborted. H_L stability verdict **PASS** for both methods.

## Setup

| Field | Value |
| --- | --- |
| Model | `google/gemma-4-31B-it` (bf16) |
| Methods | `lopi_default`, `caa` |
| Seeds | 0, 1, 2 |
| Turns per cell | 2000 |
| Checkpoints per cell | 6 (turn = 1, 50, 200, 500, 1000, 2000) |
| Cells | 2 methods × 3 seeds × 6 checkpoints = **36** |
| Hardware | spark1 GB10 (CUDA, 119 GB unified mem) |
| Adapter | gemma3 (`arch_adapter.GEMMA3`), q_norm + k_norm + v_norm (`v_proj` 4-head ratio fix) |
| Probe set | `experiments/L_marathon/probes_8.jsonl` (n=8) |
| Injected facts | `experiments/L_marathon/facts_3.jsonl` (n=3) |
| Filler corpus | `experiments/L_marathon/filler.txt` |

## Verdict (`aggregate.py`)

```
n_cells_total        = 36
n_cells_aborted      = 0
n_cells_non_aborted  = 36
n_h_l_pass           = 2 / 2  (both methods)
```

| Method | Median Δ NLL (turn 2000 − turn 1) | 95 % CI | n_pairs | p (Wilcoxon) | turn-1 NLL median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lopi_default` | **0.0** | [0.0, 0.0] | 3 | 1.0 | 16.869 |
| `caa`          | **0.0** | [0.0, 0.0] | 3 | 1.0 | 16.883 |

Per-cell `nan_inf_count = 0`. No `mem_rss_jump_gt_1gb`, no `nll_decay_gt_10x`, no `alpha_zero_drift_witness` aborts.

## Interpretation

Both `lopi_default` and `caa` are **bit-stable on the held-out probe set** across 2000 turns of injected filler context — i.e. the bank is populated once at turn 0 and the probe NLL is identical at every checkpoint thereafter. This is the expected behaviour given:

1. The probe set is fixed and decoded with the same K/V cache state (no streaming chat history).
2. The bank is append-only and never compacts during the run.
3. `α=0` redline holds — no adapter parameter mutates between checkpoints.

The result is **strong but boring**: zero drift on a fixed probe set is a lower bound on stability, not a measure of recall under interactive workloads. Per `papers/v0_8_rigor_overhaul/stability_metrics.md`, v0.8 will replace this with `metrics_v2`:

- per-turn factual recall (CounterFact-1k subset, decoded at every checkpoint),
- KL-vs-base divergence on a held-out distractor set,
- residual trajectory norm tracked end-to-end.

The current `residual_norm_mu = 610.82` (constant across all 36 cells) confirms the residual stream is also unchanged — consistent with the α=0 / append-only invariants and the patcher’s `q_proj`/`k_proj`/`v_proj` post-projection hook semantics.

## Cross-arch status (sibling runs)

| Architecture | Status | Notes |
| --- | --- | --- |
| `gemma-4-31B-it`     | ✅ 36/36 cells | this report |
| `Qwen3.6-27B`        | ❌ aborted at patcher init | linear-attention; `AttnNativePatcher` cannot locate dense decoder layers (expected per `DECISIONS.md` D4) |
| `gpt-oss-120b`       | 🟢 in progress on spark1 | `runs/L1_full_v1_gpt_oss_120B/` |
| `Qwen3-4B-Instruct-2507` | ⏳ queued for v0.8 cross-arch replication |
| `Llama-3.1-8B-Instruct`  | ⏳ queued for v0.8 cross-arch replication (download in progress, unsloth mirror) |

## Reproduction

```bash
ssh spark1
cd /home/gabira/projects/RCV-HC && source .venv-gb10/bin/activate
METHODS="lopi_default caa" SEEDS="0 1 2" TURNS=2000 \
    bash scripts/dispatch_L1_spark.sh   # gemma-only run finishes in ~85 min
```

Aggregation:

```bash
cat runs/L1_full_v1_gemma4_31B/{lopi_default,caa}/seed*/cells.jsonl > /tmp/cells.jsonl
python experiments/L_marathon/aggregate.py \
    --cells /tmp/cells.jsonl \
    --out runs/L1_full_v1_gemma4_31B/aggregate_summary.json
```

## Files

- `aggregate_summary.json/summary.json` — H_L verdicts + bootstrap CIs
- `aggregate_summary.json/flat_table.csv` — per-checkpoint long table
- `lopi_default/seed{0,1,2}/cells.jsonl` — raw checkpoints
- `caa/seed{0,1,2}/cells.jsonl` — raw checkpoints
