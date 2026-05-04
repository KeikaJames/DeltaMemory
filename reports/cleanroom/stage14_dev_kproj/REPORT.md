# Stage 14 Dev Sweep — with trained InfoNCE K-projector

**Date**: Phase D (dev only)
**Model**: google/gemma-4-E2B, MPS, bf16
**Split**: dev (N=33 facts, 6 paraphrase queries each = 198 queries)
**Seeds**: 0, 1, 2
**K-projector**: `reports/cleanroom/stage14_kproj/k_projector.pt`
  (520 pairs, 104 train facts, 8 epochs, loss 2.89→0.95)

## Results (recall@1)

| Condition | seed=0 | seed=1 | seed=2 | mean |
|---|---:|---:|---:|---:|
| B0 no_memory (kproj attached, α=0) | 0.3535 | 0.3535 | 0.3535 | 0.3535 |
| **v3_period_kproj (FROZEN)** | **0.4343** | **0.4343** | **0.4343** | **0.4343** |
| v3_address_kproj | 0.2929 | 0.2980 | 0.2929 | 0.2946 |
| v3_multi_kproj | 0.3838 | 0.3889 | 0.3889 | 0.3872 |
| v3_multi_kproj + τ=0.5 | 0.0051 | 0.0051 | 0.0051 | 0.0051 |

(Numbers are deterministic across seeds because the eval pipeline has no
sampling — seeds only permute fact ordering, which the bank treats
order-invariantly.)

## Statistical test (v3_period_kproj vs B0_no_memory, per-fact paired)

| metric | value |
|---|---|
| Δrecall@1 (mean) | **+0.0808** |
| wins / losses / ties | 15 / 5 / 13 |
| Wilcoxon signed-rank p (one-sided) | **0.0122** |
| Bootstrap 95% CI for Δrecall | **[+0.010, +0.152]** |
| Cohen's d | 0.39 |

CI does not contain 0 → significant at α=0.05.

## Key takeaways

1. **InfoNCE K-projector is the linchpin.** Without training the
   projector, every bank-attached condition collapsed to recall@1 ≈ 0.000
   on N=33 facts (see `reports/cleanroom/stage14_dev/REPORT.md`).
   With training: v3_period beats B0 by **+8.1pp**, statistically
   significant.
2. **Period capture beats address capture.** Address capture lost 5.9pp
   to B0 — the K-trajectory at the address span (write-time) is too far
   off the Q-trajectory at the period (read-time).
3. **Multi (period+address) is intermediate.** Doubling the bank size
   without improving alignment dilutes the period contribution.
4. **Sharpening (τ<1) collapses everything.** τ=0.5 amplifies the
   off-target negative scores; the bank's softmax stays close to uniform
   only for τ=1.
5. **Frozen config**: `v3_period` + trained K-projector + τ=1 + ROME off
   → `deltamemory/configs/v3_frozen.yaml`.

## What this proves vs. doesn't

- ✅ Mneme v3 produces a real, repeatable, statistically
  significant lift over the no-memory baseline on a held-out dev split
  with 33 unrelated facts in the bank.
- ❌ This does NOT yet prove v3 beats prompt-insertion / RAG / MEMIT.
  Those baselines are scheduled for Phase G on the held-out **test**
  split, which has not been touched.

## Next

- E: pull Qwen3-4B + glm-4-9b on Mac, rsync to spark.
- F: ArchAdapter for Gemma-4 / Qwen3 / Llama / GLM-4.
- G: full benchmark on test split (5 models × 7 baselines, Wilcoxon +
  Holm-Bonferroni + bootstrap CIs + double-blind judge).
