# L.1 Marathon — Qwen3-4B-Instruct-2507 — COMPLETE (v0.8 cross-arch replication)

**Status**: 36/36 cells non-aborted. H_L stability verdict **PASS** for both methods.

## Setup

| Field | Value |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507` (bf16) |
| Methods | `lopi_default`, `caa` |
| Seeds | 0, 1, 2 |
| Turns per cell | 2000 |
| Checkpoints per cell | 6 (turn = 1, 50, 200, 500, 1000, 2000) |
| Cells | 2 methods × 3 seeds × 6 checkpoints = **36** |
| Hardware | spark1 GB10 (CUDA bf16) |
| Adapter | `arch_adapter.QWEN3` — q_norm + k_norm, **no v_norm** (verified `arch_adapter.py:143-172`) |

## Verdict (`aggregate.py`)

```
n_cells_total        = 36
n_cells_aborted      = 0
n_cells_non_aborted  = 36
n_h_l_pass           = 2 / 2  (both methods)
```

| Method | Median Δ NLL (turn 2000 − turn 1) | 95 % CI | n_pairs | p (Wilcoxon) | turn-1 NLL median |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lopi_default` | **0.0** | [0.0, 0.0] | 3 | 1.0 | 10.236 |
| `caa`          | **0.0** | [0.0, 0.0] | 3 | 1.0 | 7.295 |

Per-cell `nan_inf_count = 0`. No `mem_rss_jump_gt_1gb`, no `nll_decay_gt_10x`,
no `alpha_zero_drift_witness` aborts. Residual norm `~115.75` constant
across all 36 cells.

## Replication finding (D4)

This run is the **first cross-arch L.1 replication** required by
`papers/v0_8_rigor_overhaul/DECISIONS.md` D4. With the gemma-4-31B baseline,
we now have:

| Architecture | Methods PASS | Result family |
| --- | ---: | --- |
| `gemma-4-31B-it`            | 2/2 | dense, full q/k/v_norm |
| `Qwen3-4B-Instruct-2507`    | 2/2 | dense, q/k_norm only (no v_norm) |
| `Llama-3.1-8B-Instruct`     | 2/2 | dense, no q/k/v_norm (sibling report) |
| `Qwen3.6-27B`               | n/a | linear attention — patcher cannot splice K/V (excluded by design) |

The append-only / α=0 redline holds across all three dense architectures
and across two normalization regimes (gemma-style trio vs. qwen-style
q+k vs. llama-style none). H_L stability is therefore **architecture-invariant
under the dense-attention assumption**, addressing the
"single-model artifact" reviewer concern.

## Caveats

Same as gemma-4-31B: the L.1 protocol probes a fixed held-out set, so this
result is *bit-stability of bank-augmented decoding under filler context*,
not interactive recall. v0.8 will replace with `metrics_v2`
(per-turn factual recall on CounterFact-1k, KL-vs-base, residual trajectory).

## Reproduction

```bash
ssh spark1
cd /home/gabira/projects/RCV-HC && source .venv-gb10/bin/activate
bash /tmp/dispatch_L1_v08.sh   # see runs/L1_v08_dispatch.log
# aggregate
cat runs/L1_full_v1_qwen3_4B/{lopi_default,caa}/seed*/cells.jsonl > /tmp/cells.jsonl
python experiments/L_marathon/aggregate.py \
    --cells /tmp/cells.jsonl \
    --out runs/L1_full_v1_qwen3_4B/aggregate_summary.json
```

## Files

- `aggregate_summary.json/summary.json` — H_L verdicts + bootstrap CIs
- `aggregate_summary.json/flat_table.csv` — per-checkpoint long table
- `lopi_default/seed{0,1,2}/cells.jsonl`, `caa/seed{0,1,2}/cells.jsonl` — raw checkpoints
