# Phase L4 — v3.1 K-projector dev eval (Mac MPS, Gemma-4-E2B)

**Split**: `eval/splits_v31/dev_v31.jsonl` (N=41 held-out facts, sha-locked).
**Projector**: `reports/cleanroom/stage15_kproj_v31/k_projector_gemma4.pt`
  (sha256 `2532bdc53a925abcdd770a410117fcd7a79512bfdc3b7ab4ef5c07cb97a49ced`).
**Device**: Apple Metal / PyTorch MPS, bfloat16. Seed 0.
**LLM weights**: frozen (verified by α=0 → bit-equal regression test K7).

## Recall@1

| Condition | recall@1 | vs B0 |
|---|---:|---:|
| B0 no_memory                    | 0.3512 | — |
| B1 prompt_insertion (oracle)    | 0.6366 | +0.2854 |
| B2 rag_oracle (matched-budget)  | 0.6561 | +0.3049 |
| v2 period_no_kproj (Stage 13)   | 0.0122 | −0.3390 |
| **v3.1 period_kproj (this run)**| **0.5585** | **+0.2073** |

## Reading

- v3.1 lifts **+20.7pp absolute** (+59% relative) over no-memory on a held-out
  41-fact dev set, with frozen LLM weights (no LoRA, no MEMIT).
- Gap to prompt-insertion oracle (B1): −7.8pp. v3.1 recovers ~73% of the
  prompt-insertion gain *without* putting the fact in context.
- v2 alone (no projector) collapses, confirming the projector is the load-bearing
  component, not the bank itself.
- Test split (`test_v31.jsonl`) untouched per preregistration.

## Caveat

This is one seed on one model on one device. GB10 CUDA bf16 path of the same
projector still collapses (separate numerical bug, see
`reports/cleanroom/stage15_kproj_v31/REPORT.md`). Phase L5 val2 gate next.
