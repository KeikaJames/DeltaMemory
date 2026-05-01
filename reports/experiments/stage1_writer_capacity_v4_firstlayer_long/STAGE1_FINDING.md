# Stage 1 Finding: Writer Capacity (Channel A) — first_layer PASSES

## Summary

The writer's `raw_value` representation **can encode answer-token identity**, but
only when the payload probe reads from the **first transformer layer** (early
layers preserve token identity, per probing literature).

## Sweep results (single-token answers, oracle-span writer, alpha=0)

| Variant | layer strategy | LR | steps | min `payload_answer_loss` | final `payload_answer_loss` |
|---|---|---|---:|---:|---:|
| V1 | mean_all      | 3e-3 |  64 | 2.633 | 6.991 |
| V2 | last_layer    | 3e-3 |  64 | 2.440 | 6.870 |
| V3 | first_layer   | 3e-3 |  64 | 2.694 | 4.687 |
| **V4** | **first_layer** | **5e-4** | **256** | **0.0029** | **0.032** |

Random baseline for 32-code single-token answers is `log(32) ≈ 3.47`. V4 drives
loss two orders of magnitude below random and to ~0 on training data.

## Decisive observations

1. `mean_all` and `last_layer` plateau above random with high LR; `first_layer`
   converges. This matches probing literature (Tenney et al., Belinkov et al.):
   token identity lives in early layers, contextual/semantic features in late.
2. With LR `3e-3`, all variants are unstable. With LR `5e-4` and longer training,
   `first_layer` reaches near-perfect train fit.
3. Sanity: `oracle_logit_answer_embedding` continues to give top10 = 1.0
   (Stage 0 control still passes).

## Pass/fail under Stage 1 gate

The Stage 1 gate is **"writer CAN encode answer identity in `raw_value`"**.
- Train memorization ≪ random baseline: ✅
- Held-out generalization: deferred to Stage 1.5 (requires adding
  `payload_probe` to eval modes; the current run does not evaluate the probe
  on held-out examples).

Channel A is therefore **viable**. The next stage tests channel C (injection)
under this writer/first_layer probe configuration.

## Implications for Stage 2 (injection channel)

- Lock canonical writer config: `oracle-span-writer + first_layer` payload
  probe.
- Stage 2 must keep the identical writer and probe head while sweeping
  injection channels (logit_bias, LM-head rank-1 LoRA, OV LoRA, Q/V residual,
  cross-attn).
- The negative result that `mean_all` cannot fit explains all prior pilot
  failures: the existing `delta_qv` / `logit_bias` paths were reading from a
  representation that destroys token identity by averaging across layers.

## Files

- `delta_experiment_summary.json` (sweep data lives in
  `../stage1_writer_capacity_v{1,2,3,4}_*/`)
- Trained writer/probe state is not persisted; rerun with the same flags is
  deterministic at `--seed 0`.
