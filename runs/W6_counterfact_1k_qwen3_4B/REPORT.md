# W.6 CounterFact-1k AttnNativeBank — Qwen3-4B-Instruct-2507

**Status**: ✅ COMPLETED, 2026-05-07 04:55 CST (132.7 min on GB10)
**Cells**: 9,684 / 9,684 PASS (807 real prompts × 2 methods × 2 alphas × 3 seeds + 193 sentinel-dropped)
**Dataset**: `experiments/datasets/counterfact_1k.jsonl` — seed=42 sample of `azhx/counterfact` (19,728 entries → 1,000 entries; 193 dropped due to LAMA-template coverage gap).
**Hardware**: GB10 (Spark1) CUDA bf16
**PREREG**: W.6 v1 + `--counterfact-path` extension (commit e835cb6d)

## Headline

| method | α | recall@1 | margin (nats) | KL drift |
|---|---:|---:|---:|---:|
| none | 0.0 | 0.145 | −4.91 | 0.000 |
| none | 1.0 | 0.145 | −4.91 | 0.000 |
| lopi_default | 0.0 | 0.145 | −4.91 | 0.000 |
| **lopi_default** | **1.0** | **0.446** | **−0.46** | **0.46 ± 0.03** |

`recall@1` = fraction of prompts where `nll(target_new) < nll(target_true)`.
`margin = nll_true − nll_new` (positive ⇒ model prefers `target_new`).

## What this shows

1. **+30.1 pp recall lift** — AttnNativeBank (LOPI default, α=1.0) on
   Qwen3-4B-Instruct-2507 lifts recall@1 from 14.5% to 44.6% on the
   CF-1k subset. Baseline strongly prefers `target_true` (margin =
   −4.91); injection moves the model 4.45 nats toward neutrality
   (margin = −0.46).

2. **Bit-equal α=0 redline holds** — `lopi_default` at α=0 and `none`
   produce identical `nll_new`, `nll_true`, and 0.0 KL drift. The
   LOPI gate is a true no-op at α=0 (3 seeds × 807 prompts = 2,421
   row-equality check).

3. **Unrelated-window drift is bounded** — α=1.0 mean JS over 16
   unrelated windows = 0.46 ± 0.03 nats (matches the SCAR-smoke order
   of magnitude). No catastrophic off-axis interference.

4. **Per-seed identity** — recall@1 has σ=0.0 across seeds because
   the W.6 LOPI calibration is deterministic for a fixed n_calib.
   The variance comes from the per-prompt `kl_unrel` sampling
   (σ=0.03 nats).

## Caveats

- Not a head-to-head with ROME/MEMIT. This is a *mechanism* paper —
  the role of CF-1k is to confirm that ANB's "injectability" is not a
  synthetic-distractor artifact (X.7-NL) but transfers to a standard
  KE benchmark. Direct comparison with ROME/MEMIT via EasyEdit is
  parked for v0.8 / paper v2 (see Section 10.4 of the paper).
- 19.3% of prompts (193 / 1000) had relations not present in the
  LAMA-T-REx template map and are emitted as sentinel rows; they do
  not enter the recall@1 / margin numerator. Future work: extend the
  template map.
- One model (Qwen3-4B). Cross-arch CF-1k replication on Llama-3.1-8B
  and Gemma-4-31B is queued for v0.8.

## Files

- `cells.jsonl` — 9,684 raw rows (per cell: nll_new, nll_true, kl_unrel,
  status, redline_violation, …)
- `aggregate_summary.json` — cross-seed (method, α) summary
- `env.json` — runtime stamp (torch, transformers, dataset SHA-1, etc.)
- `run.log` — verbose per-progress dispatcher log

## Reproduce

```bash
# build CF-1k subset (one-off, on hf-mirror.com network)
HF_ENDPOINT=https://hf-mirror.com python scripts/build_cf_1k.py

# run (132 min on GB10)
python experiments/W6_counter_prior/run.py \
  --device cuda --dtype bfloat16 \
  --models /path/to/Qwen3-4B-Instruct-2507 \
  --methods none lopi_default \
  --alphas 0.0 1.0 \
  --seeds 0 1 2 \
  --n-prompts 1000 --n-unrelated 16 \
  --counterfact-path experiments/datasets/counterfact_1k.jsonl \
  --allow-high-drop \
  --out runs/W6_counterfact_1k_qwen3_4B/cells.jsonl

# aggregate
python scripts/aggregate_w6_counterfact.py runs/W6_counterfact_1k_qwen3_4B
```
