# Phase Q — mHC-Mneme Flagship Verification: Preregistration

**Frozen**: 2026-05-04
**Branch**: stage13-attn-native
**Commit**: 91276a09 (post Codex P2 fixes from PR #5)

---

## 1. Scope

Prove on 5 frozen flagship LLMs that:
1. The mHC shield (bank-columns-only column cap) makes α tuning insensitive —
   a single α value yields safe injection across families (the v3.1 pain point
   was a 20× spread).
2. Mneme can implant counter-prior false facts into model-generated text
   at chat-format evaluation quality.

## 2. Five Hypotheses

| ID | Hypothesis | Verification Phase | PASS criterion |
|---|---|---|---|
| **H1** | mHC shield keeps α-injection NLL drift ≤ 0.5 nats at α ∈ [0.05, 10] across all 5 models | Q2 | shield ON neutral-NLL drift ≤ 0.5 nats on ≥ 5/7 α per model, 5/5 models |
| **H2** | mHC shield preserves counter-prior lift > 0 at ≥ 5/7 α points per model | Q2 | shield ON lift > 0 on ≥ 5/7 α per model, 5/5 models |
| **H3** | Counter-prior facts can be implanted into generated text with ≥ 60% "accurate implant" rate | Q3 | ≥ 60% accurate on ≥ 3/5 subject models, judge κ ≥ 0.6 |
| **H4** | α = 0 bit-equality holds for all 5 models (red line invariant) | Q1 | 5/5 max-abs-diff = 0.0 |
| **H5** | Shield does not degrade neutral-fact baseline coherence by > 5% | Q3/Q4 | coherence drop ≤ 5% + no 5-gram contamination |

## 3. Model Array

| # | Model | HF ID | Family | Adapter | Load | Notes |
|---|---|---|---|---|---|---|
| 1 | Gemma-4-E2B | google/gemma-4-E2B | Gemma | gemma4 | bf16 ~9.6GB | v3.1 baseline |
| 2 | Qwen3-4B | Qwen/Qwen3-4B-Instruct-2507 | Qwen | qwen3 | bf16 ~7.6GB | α=0.05 in v3.1 |
| 3 | DeepSeek-R1-Distill-Qwen-32B | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | Llama | llama | bf16 ~64GB | 32B prior stress |
| 4 | GLM-4-9B | THUDM/glm-4-9b-chat | GLM | glm4 | bf16 ~18GB | ChatGLM custom-code |
| 5 | Gemma-4-31B | google/gemma-4-31B-it | Gemma | gemma4 | bf16 ~62GB | Dense flagship |

**Excluded** (needs MoE adapter + GB10 cannot host): Qwen 3.6-35B-A3B, Llama 4 Scout, DeepSeek V3/V4,
GLM-5.1, Kimi K2.6.  Documented as v3.3 future work.

## 4. Datasets

- **Q2 neutral NLL drift**: Wikitext-2 validation, 32 × 512-token non-overlapping segments
  (seed=0, sha-locked via `_wikitext2_segments`).
- **Q2/Q3 counter-prior facts**: 5 FALSE_FACTS extended to 60 facts from LAMA-TREx +
  ConceptNet + hand-crafted.  SHA to be locked before Q2 launch.
- **Q3 adversarial chat**: 60 counter-prior + 60 neutral (control) facts.  SHA to be locked
  before Q3 launch.
- **Q4 5-gram contamination**: RedPajama public subset or C4 validation subset proxy.

## 5. Per-Phase PASS Criteria & 3-Strike Rule

### Q1 — Environment Smoke
- Load all 5 models, α=0, empty bank: max-abs-diff = 0.0 (256 tokens × vocab).
- FAIL → debug adapter per failure-mode table.
- 3 strikes → `git reset --hard <Q0-commit>` + `INCIDENT.md`.

### Q2 — α-Safety NLL/Lift Sweep
- 5 models × 7 α ∈ {0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0} × shield ∈ {off, on} × 3 seeds
- PASS: shield ON NLL drift ≤ 0.5 nats AND lift > 0 at ≥ 5/7 α, per model, 5/5 models.
- Rampup (if PASS): bank_size 32 → 128 rerun.

### Q3 — Adversarial Chat Implant
- 60 counter-prior + 60 neutral facts × 5 subject models × shield ON α=1.0
- Gemma-4-31B local judge × 2 runs (κ ≥ 0.6 required)
- Labels: accurate_implant / partial_implant / not_implanted / garbled
- PASS: ≥ 60% accurate on ≥ 3/5 subjects + neutral coherence drop ≤ 5%
- Rampup (if PASS): long-prompt (256 tokens noise prefix) → multi-fact (8 facts in bank) → adversarial follow-up ("Are you sure?")

### Q4 — Statistics & Robustness
- Bootstrap 95% CI (n=10000) + Wilcoxon signed-rank paired + Holm-Bonferroni (m=5)
- 5-gram contamination flag (hit rate > 5% → exclude from primary, report in ablation)
- Layer-norm energy curves: shield ON vs OFF at α=10
- PASS: H1-H3 p < 0.01 after Holm-Bonferroni correction

### Q5 — Final Aggregation
- REPORT.md with preregistration ↔ empirical mapping
- 5 figures: α-NLL heatmap, α-lift heatmap, implant rate bars, judge confusion matrix, layer-norm curves
- PR description update with Phase Q results

## 6. Red Lines (inherited from v3.1 + v3.2)

- LLM weights frozen; α=0 bit-equal sanity gate before every run
- Any phase failure must NOT adjust test conditions for cosmetic improvement
- All raw artefacts committed to `reports/cleanroom/flagship_v32/` with per-attempt directory
- Commit as KeikaJames <gabira@bayagud.com>, no co-author trailer
- Banned word: "honest/诚实" — use "strict/preregistered/explicit"

## 7. Attempt Directory Convention

```
reports/cleanroom/flagship_v32/{phase}/{model}/attempt_{N}/
├── cmd.sh
├── stdout.log
├── stderr.log
├── result.json          # PASS/FAIL + key metrics
└── env.json             # torch / transformers / cuda / commit sha
```

## 8. Amendment Log

| Date | Amendment | Reason |
|---|---|---|
| 2026-05-04 | Initial freeze | — |
