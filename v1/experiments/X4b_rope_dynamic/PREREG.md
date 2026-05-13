# X.4b — Dynamic RoPE Consistency Witness

**Status**: locked
**Version**: `X4b.v1`
**Authored**: v0.5 cycle (Phase X follow-up)
**Owner**: opus

## 1. Question

User concern (疑问②, paraphrased):
> "Different time-step injected memory — how do their relative positions stay
> consistent? At seq-len 16k / 32k, do accumulated position-encoding errors
> kill early memories?"

Claim under test (Mneme v0.4 design):
> AttnNativeBank captures bank K *pre-RoPE*; at read time, bank K is rotated
> using the *current query's* `cos/sin` (`adapter.apply_rope(q_pre, k_pre,
> cos, sin)`). Therefore `q_post · k_post` is invariant to the absolute
> position of the query — bank score should not drift as the query moves
> further along the sequence.

We test this dynamically: **same fact, same bank, vary the absolute
position of the read query** by prepending filler tokens, and measure
top-1 score margin and target_new probability on a held-out probe.

## 2. Hypotheses

* **H_X4b.0 (red-line, must hold)**: at α = 0, the model's logits at
  position-shifted read prompts equal the unpatched model's logits at the
  same shifted prompt, bit-for-bit. Tolerance: max-abs-diff ≤ 1e-4 in
  fp32, ≤ 5e-3 in bf16.
* **H_X4b.1 (main)**: at α = 1, the bank-injected score margin
  `s(target_new) − s(target_canonical)` at the read step is
  position-stable: max deviation across read positions
  `P ∈ {16, 64, 256, 1024, 4096}` is bounded by:
    - σ ≤ 0.10 (relative, fp32)
    - σ ≤ 0.25 (relative, bf16)
* **H_X4b.2 (negative control)**: a model variant with **post-RoPE** bank
  K capture (ablation A1) shows σ ≥ 1.0 — i.e. the design choice is
  load-bearing; if A1 ablation does not break invariance, the claim is
  vacuous.

## 3. Grid

| factor | levels |
|---|---|
| model | `Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B` |
| dtype | fp32 (primary), bf16 (cross-precision check) |
| seed | 0, 1, 2 |
| read-position P | 16, 64, 256, 1024, 4096 |
| facts | 4 hand-crafted counterfactuals (subject, relation, target_new, target_canonical) |
| arm | `static-pre-rope` (default), `static-post-rope-A1` (ablation, P_subset = {16, 1024}) |

Total cells: 2 × 1 × 3 × 5 × 4 × 1 = 120 (default arm) + 2 × 1 × 3 × 2 × 4 × 1 = 48 (ablation) = **168 cells**.

## 4. Procedure (per cell)

1. Load model, install AttnNativePatcher, fresh_bank.
2. Write fact via `write_fact(write_prompt=fact_line, address=subject)`.
3. Build read query: `filler(P) + " " + read_prompt` where filler is
   deterministic ("the the the …" capped to P tokens after tokenization).
4. Forward unpatched model → record logits at last position.
5. Forward with bank, α=0 → record logits, verify red-line.
6. Forward with bank, α=1 → record logits.
7. Score margin = `logits[id(target_new)] − logits[id(target_canonical)]`.
8. Append cell row to `cells.jsonl`.

## 5. Authenticity

Per `docs/authenticity.md`. `env.json` written via
`tools.env_writer.write_env_json`. Per-cell row keyed by
`cell_id = sha1(model|seed|alpha|P|fact_id|arm)[:16]`. Resume-safe.
No editorial trimming; raw cells retained.

## 6. Out of scope

* Multi-fact bank scaling (covered by X.1).
* RoPE base-frequency interpolation (a separate concern; this PREREG
  uses each model's native max-position).
* Position scaling at P > native context length (would require RoPE
  scaling — left for future work; cells with P > model.config.max_position_embeddings
  are skipped with `status="skipped_oom_or_overflow"`).

## 7. Abort conditions

* Any unpatched-vs-α=0 diff exceeding tolerance → red-line fail; investigate before reporting α=1 results.
* Any NaN/Inf in logits → status="nan_inf"; do not aggregate that cell.

## 8. Deliverable

* `experiments/X4b_rope_dynamic/cells.jsonl` (raw)
* `experiments/X4b_rope_dynamic/summary.json` (aggregate from `aggregate.py`)
* `experiments/X4b_rope_dynamic/REPORT.md` (verdict + figure)
* `docs/figures/x4b_rope_dynamic_margin_vs_pos.png`
