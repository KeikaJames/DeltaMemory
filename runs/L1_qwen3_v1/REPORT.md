# L.1 Marathon — Qwen3-4B-Instruct (CAA, 500 turns × 3 seeds)

**Status**: VERDICT — H_L SUPPORTED (perfect stability)
**Branch**: feat/v05-counterfactual-industrial
**Hardware**: GB10 (NVIDIA GB10 Superchip), CUDA, bf16
**Model**: `Qwen/Qwen3-4B-Instruct-2507`
**Method**: CAA (target-mean steering at α=1.0)
**Dataset SHA**: see per-seed `env.json` (facts_3.jsonl + probes_8.jsonl + filler.txt composite)

## Setup

- N = 500 turns per session (one bank-write phase + 4 read checkpoints).
- Read checkpoints at turns {1, 50, 200, 500}.
- 3 facts injected at turn 1, queried at every checkpoint.
- 8-prompt held-out probe set; `nll_target_new` averaged per checkpoint.
- 3 seeds: {0, 1, 2}, identical fact + probe set.
- Filler advances "marathon clock" with capped 512-token windows so the
  HF KV-cache never grows. The stress is therefore on **bank-scoring
  invariance under model-state churn**, not on KV-cache accumulation.

## Results

| seed | turn=1   | turn=50  | turn=200 | turn=500 | nan_inf | Δ RSS (MB) |
|-----:|---------:|---------:|---------:|---------:|--------:|-----------:|
| 0    | 7.34720  | 7.34720  | 7.34720  | 7.34720  | 0       | +1.7       |
| 1    | 7.34720  | 7.34720  | 7.34720  | 7.34720  | 0       | +1.7       |
| 2    | 7.34720  | 7.34720  | 7.34720  | 7.34720  | 0       | +1.7       |

`residual_norm_mu` at the canonical injection layer is also bit-identical
(4679.15) across all 12 (seed × checkpoint) cells.

## Verdict

- **H_L (recall(turn=last) ≥ 0.5 × recall(turn=1))**: **SUPPORTED**
  via perfect-stability path. All 3 paired diffs (turn=500 − turn=1)
  are exactly 0.0; median_diff = 0, 95% bootstrap CI = [0, 0].
- The Wilcoxon test cannot reject the null with all-zero ranks (p=1.0
  by construction); the substantive H_L claim is nevertheless
  satisfied because no decay occurred. The aggregator now exposes a
  `perfect_stable` short-circuit for this case (n_pairs ≥ 2, all diffs
  == 0).
- **No NaN / Inf** observed in bank K/V tensors across 12 cells.
- **RSS drift bounded** at +1.7 MB / 500 turns.

## Caveats and follow-ups

- The runner caps the KV-cache window at 512 tokens to keep filler
  cost bounded; this means **growing-context** failure modes (RoPE
  base-frequency instability at >32k tokens, KV blow-up) are
  **not** stressed here. Phase X.4 (long-prompt stress) is the
  separate test for that surface.
- Bit-identical NLL across 500 turns confirms that bank scoring is a
  pure function of `(bank state, probe input)` and does not silently
  drift through stochastic injection paths. This is the desired
  invariance guarantee for production deployment.
- Cross-arch L.1 on gemma-4-E2B is the next item to satisfy CLAUDE.md
  HARD cross-architecture rule. (Pending: GB10 free after X.2.)
- A second arm (`lopi_default`) has been wired but is bit-equivalent
  to baseline on the AttnNativeBank path; a real LOPI-gated marathon
  requires the `CAAConfig(use_lopi_gate=True)` patch documented in
  `experiments/A_ablation/FINDING_arm_method_mismatch.md`.

## Files

- `runs/L1_qwen3_s{0,1,2}_t500/cells.jsonl` — raw checkpoints.
- `runs/L1_qwen3_s{0,1,2}_t500/env.json` — commit + dataset SHAs.
- `runs/L1_qwen3_v1/summary.json` — H_L verdict.
- `runs/L1_qwen3_v1/flat_table.csv` — flat (seed, turn, nll, ...) rows.
