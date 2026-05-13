---
audit_item: A5
verdict: partial
evidence_path: experiments/A5_profiler_n100/REPORT.md
---

# A5 — Profiler evidence audit

## Diagnosis

The complaint is partly correct. The old S-7 evidence used N=8 prompts and had a mixed downstream verdict:

- Qwen2.5-0.5B-Instruct: auto drift `<` static drift.
- Qwen2.5-1.5B: auto drift `>` static drift at α=2 and α=4.

Therefore U-LOPI should not be a headline quality claim. It is safer as an ablation/calibration flag.

## New evidence

The N=100 profiler rerun used Wikitext-2 validation with a fixed corpus artifact. Results:

- Qwen2.5-0.5B: `mu_arch=5`, matching N=8.
- Qwen2.5-1.5B: `mu_arch=5`, matching N=8.
- Four topic buckets all selected layer 5 for both models.

This strengthens the narrow statement: “the residual-variance profiler’s layer argmax is stable on these Qwen2.5 models under this corpus.”

## Remaining open point

Layer-argmax stability is not the same as downstream quality. The old drift evidence remains mixed, and no N=100 downstream α sweep was run here. Main-plan impact: demote U-LOPI from a primary mechanism claim to an ablation/calibration flag until a larger task-level sweep shows consistent drift reduction.
