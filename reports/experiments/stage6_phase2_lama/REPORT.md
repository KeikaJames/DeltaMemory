# Stage 6 Phase 2 — LAMA `factual_capital_binding` end-to-end LM-head LoRA

**Setup.** Frozen `google/gemma-4-E2B`, MPS / Metal, bf16. Token-preserving writer (oracle-span, first-layer, attn pool) emits a payload that is consumed simultaneously by four eval channels: `delta_qv` (Q/V residual), `payload_probe` (full-vocab CE via frozen LM head), `logit_bias` (additive logit bias), `lm_head_lora` (rank-4 LoRA on LM head, scale=50). Trained 1500 steps with answer-token CE (weight=1.0), payload-embedding loss (0.5), Stage-2 swap loss in `lm_head_lora` mode (0.5, margin=2.0), and per-channel auxiliary losses (0.5 each).

Dataset: full LAMA pool (56 unique country/capital pairs). `train_samples = eval_samples = 56` ⇒ train and eval are the **same** 56 cards by design (binding test, not generalization test — the pool is too small for a meaningful disjoint split, and Stage 7A established that disjoint splits trigger a closed-vocab projector flaw, see `reports/experiments/stage7a_lama_capital/`).

3 seeds (0,1,2), `--writer-pools attn --swap-options on`.

## Results — eval channels (mean ± std across 3 seeds)

| channel | top1 | top10 | answer NLL | rank |
| --- | ---: | ---: | ---: | ---: |
| `no_memory` (baseline, no payload) | 0.000 ± 0.000 | 0.000 | 17.16 | 12354 |
| `oracle_logit_answer_embedding` (upper bound) | 0.964 ± 0.000 | 0.982 | 0.79 | 42.6 |
| `payload_probe` | **1.000 ± 0.000** | 1.000 | 0.03 | 1.0 |
| `logit_bias` | 0.964 ± 0.000 | 1.000 | 0.20 | 1.0 |
| `lm_head_lora` (rank-4) | **1.000 ± 0.000** | 1.000 | 0.00 | 1.0 |
| `delta_qv` (Q/V residual) | 1.000 ± 0.000 | 1.000 | 0.00 | 1.0 |

Strict Stage 6 gate (`top1 ≥ 0.85` on `payload_probe` / `logit_bias` / `lm_head_lora`): **PASS** on all three channels, all three seeds, with `no_memory` baseline at 0.0 (no leakage).

## Results — swap controls (binding specificity)

| control | binding margin (foreign − correct NLL) | paired-answer rate |
| --- | ---: | ---: |
| `lm_head_lora_oracle_correct` | +23.20 | correct = 1.000 |
| `lm_head_lora_oracle_paired` | **−9.56 ± 1.70** | **paired = 0.506 ± 0.008** |
| `lm_head_lora_correct_address_paired_payload` | +2.71 | paired = 0.500 ± 0.000 |
| `logit_bias_oracle_paired` | −24.70 | paired = 0.482 |

Reading: when the payload is swapped to a paired (foreign) card's payload, the LoRA output flips to the foreign answer ~50% of the time. The strict swap gate (`paired ≥ 0.8`) is **not** met — the LoRA partially binds payload→answer but is not yet payload-specific enough to override the address-conditioned default.

## What this proves vs. does not prove

**Proves.**
1. The answer-token CE + LM-head rank-4 LoRA pipeline reaches the upper-bound oracle channel on LAMA factual binding (`top1 = 1.000` vs. oracle 0.964) under in-distribution evaluation.
2. The previous synthetic-data wall (Stage 2C `lm_head_lora` top1 ≈ 0.375; Stage 7A linear probe held-out 0.094) is **task-specific, not architecture-specific** — the identical pipeline applied to LAMA factual data trivially solves the binding when the underlying pretrained model has matching factual structure.
3. `no_memory = 0.000` confirms the address tokens (`ADDR::country::France`) are opaque to the frozen base — Gemma cannot answer without the in-context payload, ruling out trivial leakage.

**Does not prove.**
1. *Generalization to novel facts.* Train and eval share the same 56 cards. With only 56 unique LAMA pairs available, a meaningful disjoint split is not feasible (and Stage 7A documents the closed-vocab projector flaw for disjoint splits).
2. *Strict payload binding.* Paired-flip rate of ~0.5 indicates the LoRA encodes a mixture of the correct answer and the address-conditioned default; full payload-specific binding would need flip rate ≥ 0.8.

## Next steps

- **Stronger swap supervision.** Increase `stage2_swap_loss_weight` from 0.5 → 1.5–2.0 and/or extend warmup, to push paired-flip ≥ 0.8 without losing in-distribution `top1`.
- **Larger factual pool.** Replace the 56-pair LAMA capital subset with the full LAMA-UHN or T-REx slice (~1k+ pairs) so a disjoint-country split becomes statistically meaningful and we can report true held-out generalization.
- **Channel ablation.** Remove `delta_qv` to confirm `lm_head_lora` alone carries the signal; remove `payload_probe` aux loss to confirm `lm_head_lora` doesn't free-ride on the full-vocab CE shadow head.

Reports per-seed: `phase2_pool-attn_swap-on_seed-{0,1,2}/delta_experiment_summary.json`.
