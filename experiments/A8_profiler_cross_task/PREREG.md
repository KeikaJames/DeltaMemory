# A.8 Profiler Cross-Task Generalization — PREREG

## Status

`prereg_version: A8.v1`
`locked_at: 2026-05-06`
`branch: feat/v05-counterfactual-industrial`

## Question

User 疑问③:

> one-shot residual profiler 假设"残差范数分布主要由架构决定"，但：
>
> 1. 不同任务（代码、数学、对话、创意）的层重要性差异极大。
> 2. 当前测试主要在小模型（Qwen2.5-0.5B）上，大规模模型 + 真实复杂任务下的表现如何？

The implicit claim of `lopi_profiler.profile_residuals` is that
`mu_arch = argmax(sigma_base)` is **architecture-determined** (i.e.
mostly invariant to the profile corpus) and therefore the default
neutral corpus is sufficient.

This phase tests that claim head-on by *forcing* the profile corpus
to vary across four task domains, on a ladder of model scales.

## Hypotheses (paired across (model, task))

* **H_A8.0 (red-line)** — `profile_residuals` is a pure observation;
  weights are bit-equal pre/post on every (model, task) cell.
  Verified by hashing `model.state_dict()` before/after each call.
  Failure ⇒ abort — the profiler has a side effect.

* **H_A8.1 (intra-model task-invariance of mu_arch)** — within a
  given model, `mu_arch` chosen on each of {code, math, dialog,
  creative} corpora differs by **at most 2 layers** (i.e. for each
  model, `max(mu_arch) − min(mu_arch) ≤ 2`).
  *Supported* if true on ≥ 4/5 models.

* **H_A8.2 (default-corpus is a fair representative)** — for every
  (model, task) cell, `|mu_arch_default − mu_arch_task| ≤ 2`.
  *Supported* if true on ≥ 18/20 (model, task) cells.

* **H_A8.3 (sigma_base ranking is task-stable)** — Kendall-τ between
  `sigma_base` rankings on the default corpus and each task corpus
  ≥ 0.6 on every (model, task) cell.
  *Supported* if true on ≥ 16/20 cells.

* **H_A8.4 (inter-model variance is real)** — within the *same*
  corpus, `mu_arch` differs across models. This is **expected
  behaviour** (the profiler is supposed to adapt per architecture);
  we record it as documentation, not as a falsifier.

If H_A8.1 + H_A8.2 + H_A8.3 all hold, the profiler is *empirically*
architecture-determined and the default corpus is defensible.
If any of the three fails, the profiler is **task-sensitive** and
we ship a remediation track in A.8b (per-task profile cache).

## Models

**Two-tier flagship-class ladder.** No toy models (gpt2 / Qwen2.5-0.5B
etc are explicitly excluded).

### Tier A — MPS-tier (64 GB dev box, bf16, run *now*)

| short | name                              | scale   | family       |
|-------|-----------------------------------|---------|--------------|
| g31b  | `google/gemma-3-1b-it`            | 1B      | gemma3       |
| g22b  | `google/gemma-2-2b`               | 2.6B    | gemma2       |
| qw34  | `Qwen/Qwen3-4B-Instruct-2507`     | 4B      | qwen3        |
| glm9  | `THUDM/GLM-4-9B-0414`             | 9B      | glm4         |

Note: `google/gemma-4-E{2,4}B` are also cached but require a
transformers version newer than 5.2.0 (multimodal
`Gemma4ForConditionalGeneration`). `THUDM/glm-4-9b-chat` ships a
vendored modeling file that's incompatible with transformers 5.2.0
(`ChatGLMConfig` missing `max_length`). Both are deferred to a
future transformers bump. The current 4-model ladder covers four
distinct architectures (gemma3, gemma2, qwen3, glm4) at 1B / 2.6B /
4B / 9B scales.

### Tier B — GB10-tier (128 GB CUDA, bf16, run on GB10 box)

| short    | name                                     | scale     | family       |
|----------|------------------------------------------|-----------|--------------|
| ds32     | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 32B     | qwen2-distill|
| qw35moe  | `Qwen/Qwen3.5-35B-A3B-Base`              | 35B(A3B)  | qwen3-MoE    |

Tier A runs immediately on the Mac box (this PR). Tier B runs on
GB10 in Phase D.6 with identical PREREG and identical corpora; results
land in the same `cells.jsonl` schema (resume-safe), summarised under
`runs/A8_full_gb10/`. Cross-architecture coverage across both tiers:
gemma3, gemma3n (matformer), qwen3, qwen3-MoE, glm4, qwen2-distill =
six families.

Device/dtype: `mps` + `bf16` for Tier A (validated stack); `cuda` +
`bf16` for Tier B. Aggregate H_A8.* hypotheses are evaluated on the
combined 7-model × 5-corpus = 35-cell grid.

## Tasks (corpora)

Each corpus is **N=20 short English prompts** selected to be domain-
unambiguous. All corpora are committed in
`experiments/A8_profiler_cross_task/corpora/` with sha pinning.

* **default** — `default_profile_corpus()` from lopi_profiler (10
  zh/en mixed neutral strings; the production default).
* **code** — 20 Python/JS code-completion stubs (e.g. `def fib(n):`,
  `function quicksort(arr) {`).
* **math** — 20 GSM8K-style word problems (e.g. `If Alice has 3 apples
  and Bob gives her 2, how many ...`).
* **dialog** — 20 multi-turn chat snippets ending with the user's
  next question (e.g. `User: hi\nAssistant: hello\nUser: who am I?`).
* **creative** — 20 story-prose continuations (e.g. `The lighthouse
  keeper had not seen another soul in`).

Hand-written, not scraped, so we pin them by sha and they are
permanently reproducible.

## Procedure (per (model, task) cell)

1. Load `(tok, model)` once per model on `device=cpu, dtype=fp32`
   (small enough to fit; eliminates fp/bf16 noise as a confound).
2. `pre_hash = sha256(state_dict_bytes(model))`.
3. `profile = profile_residuals(model, tok, prompts=task_corpus,
    device='cpu', max_length=64)`.
4. `post_hash = sha256(state_dict_bytes(model))`.
5. Assert `pre_hash == post_hash` (H_A8.0); on failure write
   `status="weight_drift"` and abort the cell.
6. Record per cell: `model`, `task`, `mu_arch` (abs and fractional),
   `sigma_base[]` (full curve), `mu_base[]`, `eta_sigma`,
   `profile_corpus_sha`, `state_pre_sha256`, `state_post_sha256`.

5 models × 5 corpora = **25 cells**. Each cell is one short
forward pass; total wall-clock estimated ≤ 5 minutes on CPU.

## Aggregate (`aggregate.py`)

For each model, compute:
* `mu_arch_by_task = {task: layer_index}` and the spread `(max−min)`
  → H_A8.1 verdict.
* `delta_default[task] = |mu_arch_default − mu_arch_task|` for each
  non-default task → H_A8.2 verdict.
* `kendall_tau(sigma_base_default, sigma_base_task)` → H_A8.3 verdict.

Also write a single Markdown table per model with the 5×L `sigma_base`
curves so a reviewer can eyeball the inter-task variance.

## Authenticity

* `env.json` per run: `prereg_version=A8.v1`, dataset_sha = sha1 of
  the four corpus files, git commit, dirty hash, device, dtype, torch
  version, transformers version.
* Smoke = 2 models × 2 tasks (= 4 cells); full = 5×5 (= 25 cells).
* Cells.jsonl + summary.json + REPORT.md committed under `runs/A8_*`.

## Out of scope

* Larger models (GB10 will repeat A.8 with Qwen3-4B + Gemma-3-4b — see D.5).
* Non-English corpora (folded into Phase X.5 multi-language).
* Actually injecting at the chosen `mu_arch` and measuring downstream
  effect — that's W.6's job.
