# A.8 Profiler Cross-Task Generalization — Report (Tier A)

**PREREG**: A8.v1
**Run**: `runs/A8_full/` (4 models × 5 tasks = 20 cells)
**Device/dtype**: MPS / bf16
**Status**: H_A8.0 / H_A8.1 / H_A8.2 / H_A8.3 — **all supported**

## TL;DR

User 疑问③ asked whether the one-shot residual profiler's claim that
`mu_arch` is *architecture-determined* (and therefore the default
neutral corpus is sufficient) holds when the profile corpus is forced
to vary across task domains, on flagship-class models.

**Answer: yes, empirically, on the flagship Tier-A ladder.** Across
four architectures (gemma3, gemma2, qwen3, glm4) at 1B / 2.6B / 4B /
9B scales, the layer chosen by `argmax(sigma_base)` differs by **at
most 2 layers** between any two task corpora — well inside
`mu_arch ± 2` tolerance defined in PREREG.

## Per-model `mu_arch` table

| model                          | L  | default | code | math | dialog | creative | spread |
|--------------------------------|----|---------|------|------|--------|----------|--------|
| google/gemma-3-1b-it           | 26 | 18      | 18   | 18   | 18     | 17       | 1      |
| google/gemma-2-2b              | 26 | 23      | 23   | 23   | 23     | 23       | 0      |
| Qwen/Qwen3-4B-Instruct-2507    | 36 | 16      | 16   | 16   | 16     | 16       | 0      |
| THUDM/GLM-4-9B-0414            | 40 | 36      | 34   | 35   | 36     | 36       | 2      |

Three of four models pick **the same layer** for every task; GLM-4-9B
flickers within a 2-layer band (34–36). The default neutral corpus
matches every task corpus to within ±2 in every (model, task) cell.

## Hypothesis verdicts

| ID       | Statement                                    | Verdict      | Detail        |
|----------|----------------------------------------------|--------------|---------------|
| H_A8.0   | profiler is side-effect-free                  | **supported** | state_sha pre==post on 20/20 cells |
| H_A8.1   | intra-model spread ≤ 2 layers                 | **supported** | 4/4 models pass |
| H_A8.2   | default within ±2 of every task               | **supported** | 16/16 (model,task) pairs pass |
| H_A8.3   | Kendall-τ(σ_default, σ_task) ≥ 0.6            | **supported** | 16/16 pairs pass |

## Calibration vs the earlier toy-model finding

A prior CPU/fp32 smoke ran on `gpt2 / gpt2-medium / Qwen2.5-0.5B /
Qwen2.5-1.5B / gemma-3-1b-it`. **Qwen2.5-0.5B** showed a 3-layer
spread (default=15, code/math/dialog=12, creative=11) that would have
violated H_A8.1 had it been part of Tier A. Re-classifying it as a
toy-scale artifact: at 0.5B parameters, the residual-norm σ profile
is sufficiently flat across mid layers that small task differences
flip the argmax. Flagship-scale models (≥ 1B) show a peakier σ curve
and the argmax is stable under task variation.

This is itself a useful documented boundary: the profiler's
"architecture-determined" claim is **scale-conditional**. Below ≈1B
parameters we should use a per-task profile or fall back to the
default corpus only with a documented disclaimer; at ≥1B the default
is empirically defensible.

## What this does *not* settle

* **Tier B (32B + 35B-MoE)** runs on GB10. The two cached models
  (`DeepSeek-R1-Distill-Qwen-32B`, `Qwen3.5-35B-A3B-Base`) extend the
  ladder to 32B-class on the same PREREG. Until Tier B lands the
  scale-dependence claim is anchored at 9B.
* **Per-prompt (not per-corpus) variance.** Tier A measures across
  domain corpora, each N=20; we have not measured how much `mu_arch`
  varies within a single corpus across N=1 prompt vs N=20.
* **Downstream injection effect.** That `mu_arch` is task-stable does
  not by itself prove that injecting at the *same* layer is equally
  effective across tasks. That is W.6's job (counter-prior override
  task-grouped re-cut, future).

## Authenticity

* `env.json`: prereg_version=A8.v1, dataset_sha = sha1 dict over the
  4 corpus files (default + code + math + dialog + creative).
* `cells.jsonl`: full per-cell `sigma_base[]`, `mu_base[]`, `mu_arch`,
  `state_sha_pre`, `state_sha_post`, `elapsed_s`.
* `summary.json`: hypothesis verdicts + per-model deltas + Kendall-τ.
* All cells produced by a single `python3 run.py --device mps --dtype
  bf16`; resume-safe via `cell_id`.
