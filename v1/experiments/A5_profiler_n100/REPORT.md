---
audit_item: A5
verdict: partial
evidence_path: experiments/A5_profiler_n100/raw_cells.json
---

# A5 — U-LOPI profiler N=100 audit report

## Setup

- Corpus: `experiments/datasets/profiler_corpus_100.jsonl`
- Source: Wikitext-2 `wikitext-2-raw-v1/validation`, deterministic keyword-bucket sampling.
- Buckets: math/code/creative/news, 25 rows each.
- Device/dtype: MPS bf16.
- Script: `experiments/A5_profiler_n100/run_profiler_n100.py`.

## N=8 vs N=100 layer stability

| model | N=8 `mu_arch` | N=100 `mu_arch` | N=100 sigma argmax | stable? |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | 5 | 5 | 5 | yes |
| Qwen2.5-1.5B | 5 | 5 | 5 | yes |

The argmax layer is stable at layer 5 for both sizes.

## Task diversity

| model | math | code | creative | news | conclusion |
|---|---:|---:|---:|---:|---|
| Qwen2.5-0.5B | 5 | 5 | 5 | 5 | stable |
| Qwen2.5-1.5B | 5 | 5 | 5 | 5 | stable |

The Wikitext topic proxy did not show task-dependent layer movement. The buckets are weak labels, so this is not a final task-general proof.

## Bootstrap / CI note

The current profiler stores aggregate `sigma_base`, not per-prompt per-layer samples. The report therefore includes a conservative sensitivity bootstrap with ±5% noise around aggregate sigmas. It shows a broad plateau around layers 3–16 even though the exact argmax is 5. This supports “stable peak region” more than a sharp layer-5 theorem.

## Verdict

Partial resolution. The N=100 rerun removes the strongest small-N objection for Qwen2.5 0.5B/1.5B layer selection, but it does **not** rescue the stronger claim that auto mode always improves downstream drift. The published N=8 drift table remains mixed: 0.5B favored auto, 1.5B favored static at α≥2.
