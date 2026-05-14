# Exp35b Deviations Log

Append-only record of any deviation from `preregister.json` discovered during
execution. Each entry is committed at the moment of discovery, before any
downstream metric is computed.

---

## D2 subject-collision target (2026-XX-XX, corpus build step 00)

**Preregistered**: `subject_collision_target_frac: 0.30`

**Discovered**: The full `azhx/counterfact` dataset has only 901 multi-fact
subjects out of 18,466 unique subjects (4.9%). Even sampling 10000/19728 to
maximise multi-fact subjects, the upper bound is ~11% subject-collision frac.
At our actual sampled 10k, the realised collision frac is **6.78%**
(`corpus_meta.json`).

**Decision**: The D2 audit will run on the natural collision subset (~680
test facts whose subject collides with another bank fact). The N=10k corpus
construction is honest. We do NOT synthesise additional collisions because
that would itself be a cheat (artificial collisions != naturally collided
subjects).

**Impact on verdict**: D2 statistical power is reduced. If the natural
collision subset is too small for the pre-registered top-1 ≥ 25% threshold
to be meaningfully measurable (N < 300), we will report D2 as INCONCLUSIVE
rather than PASS/FAIL.

---

## D3 — independent_paraphrases_model substitution

**Pre-registered**: `ollama:gpt-oss:120b`
**Actual**: `ollama:qwen3-coder:30b`

**Reason**: gpt-oss:120b on this Mac runs in forced "thinking" mode (~1.6 min/call via API), which would push the 1500-fact × 2-paraphrase generation to >40 hours. qwen3-coder:30b runs at ~1 fact/s with comparable cloze-form quality on sampled inspection.

**Independence caveat**: qwen3-coder is a Qwen3 family model (same family as the eval model Qwen3-4B-Instruct-2507) but **a different model** (3x parameter count, coder-specialised, separate instruction-tuning data). Independence is therefore **partial** — strong distributional shift but not architectural independence.

**Mitigation**: D3 audit accepted with downgraded confidence label. The verdict will report D3 results with explicit "partial-independence" qualification. Empty-paraphrase facts (32/1500, 2.1%) excluded from D3 metric. Short paraphrases (<10 chars, 161/1500) flagged.

**Stats**:
- 1500/1500 facts processed
- 1339/1500 (89.3%) have 2 non-empty paraphrases
- 1468/1500 (97.9%) have ≥1 non-empty paraphrase
- 32/1500 (2.1%) fully empty (refused/leakage-rejected)
- mean paraphrase length: 50.9 chars

Recorded: 2026-05-15
