# Benchmark Results

This page is the single source of truth for **what works, what doesn't, and how confident we are**. We update it as Phase M (cross-arch v3.1) runs land.

> **Statistical convention.** All recall@1 numbers are reported with a paired Wilcoxon test against B0 (no-memory) on the same items, plus a 1000-iteration paired bootstrap 95% CI on the difference. We do NOT report mean ± SE, because the held-out fact distribution is not Gaussian.

---

## 1. Conservation law (the red line)

| model | adapter | device | dtype | max-abs-diff | bit_equal | source |
|---|---|---|---|---:|---|---|
| google/gemma-4-E2B | gemma4 | Mac MPS | bf16 | 0.000e+00 | ✅ True | `tests/conservation_real_models.py` |
| google/gemma-4-E2B | gemma4 | GB10 CUDA | bf16 | 0.000e+00 | ✅ True | (same) |
| Qwen/Qwen3-4B-Instruct-2507 | qwen3 | GB10 CUDA | bf16 | 0.000e+00 | ✅ True | (same) |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | llama | GB10 CUDA | bf16 | *pending* | *pending* | dl in progress on Mac → rsync |
| THUDM/GLM-4-9B-0414 | glm4 | GB10 CUDA | bf16 | *pending* | *pending* | dl in progress |

**Pass criterion.** `bit_equal = True` is the gate; we don't ship an adapter that fails it.

## 2. Phase N — intervention demo (qualitative, 5 facts)

Source: `transcripts/v3_intervention/`

### Gemma-4-E2B (v3 K-projector trained on this arch)

| fact | model knew? | B0 logprob | B1 prompt | v3 bank | Δ(v3−B0) |
|---|---|---:|---:|---:|---:|
| f1 mayor of Paris → Anne Hidalgo | **no** | −5.05 | −0.36 | **−0.64** | **+4.41** ≈ 80× prob |
| f2 architect of Eiffel Tower → Gustave Eiffel | yes | −0.36 | −0.52 | −0.38 | −0.02 |
| f3 painter of Mona Lisa → Leonardo | yes | −0.19 | −0.47 | −0.20 | −0.01 |
| f4 general relativity → Albert | yes | −0.20 | −0.52 | −0.16 | +0.04 |
| f5 Python creator → Guido | yes | −0.07 | −0.32 | −0.07 | −0.00 |

The bank rescues exactly the fact the model didn't know (+4.41 logprob), and stays within ±0.04 of B0 on the four it already knew. **No pollution** is the cleanest property of v3 frozen on its trained architecture.

### Qwen3-4B-Instruct (v3 K-projector NOT trained on this arch)

All 5 facts collapse ~−12 logprob. **Expected** — bank K is in Gemma-4's basis, not Qwen3's. Phase L (v3.1 cross-arch retraining) addresses this.

| fact | B0 logprob | v3 bank | Δ(v3−B0) |
|---|---:|---:|---:|
| f1 | −10.16 | −12.70 | −2.54 |
| f2 | −2.95 | −14.63 | −11.68 |
| f3 | −0.32 | −14.55 | −14.23 |
| f4 | −0.28 | −12.68 | −12.40 |
| f5 | −2.11 | −13.55 | −11.43 |

Note that the **conservation law still holds at α=0** for Qwen3 (max-abs = 0.000) — the collapse only appears when you actively inject α > 0 with the wrong projector.

## 3. Phase G — held-out test eval (frozen v3 on Gemma-4-E2B)

Source: `reports/cleanroom/stage14_test_gemma4_e2b/REPORT.md`, commit `c7ba6b5`.

| condition | recall@1 (N=39) | wilcoxon p vs B0 | bootstrap 95% CI vs B0 |
|---|---:|---:|---:|
| B0 no-memory | 0.359 | — | — |
| **v3 frozen** | **0.278** | **0.007** | **[−0.144, −0.018]** |
| B1 prompt-insertion | 0.658 | 7.6e-7 | [+0.221, +0.376] |
| v2 (no projector) | 0.000 | — | — |

**Verdict.** H1 (v3 > B0) is **rejected** (p = 0.007 in the wrong direction). The decomposition `v3 − B0 = (v3 − v2) + (v2 − B0) = +0.278 + (−0.359) = −0.081` shows the projector lifts the bank but the bank itself starts well below B0 because of softmax dilution at N=39 (39 bank entries compete on equal terms with each prompt token in softmax).

The methodology amendment to the preregistration (Stage 15) describes the three structural fixes: `bank_topk` gating (already shipped), 5× training scale + cross-relation hard negs (Phase L1), val-2 second held-out split (Phase L5).

## 4. Phase M — cross-arch v3.1 benchmark (planned)

| model | adapter | recall@1 v3.1 | recall@1 B0 | recall@1 B1 | status |
|---|---|---:|---:|---:|---|
| google/gemma-4-E2B | gemma4 | *pending* | 0.359 | 0.658 | Phase L training |
| Qwen/Qwen3-4B-Instruct-2507 | qwen3 | *pending* | *pending* | *pending* | Phase L training |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | llama | *pending* | *pending* | *pending* | dl on Mac, will rsync to GB10 |
| THUDM/GLM-4-9B-0414 | glm4 | *pending* | *pending* | *pending* | dl in progress |
| google/gemma-4-31B-it | gemma4 | *pending* | *pending* | *pending* | not downloaded |

The bench script is `scripts/run_v31_benchmark_full.py` (Phase M1). Each row reports:

- **B0** no memory (frozen LLM only)
- **B1** prompt-insertion (frozen LLM, fact in context)
- **B2** RAG over a 1-NN sentence retriever (frozen LLM, retrieved sentence in context)
- **v3.1 bank** (frozen LLM, fact written into attn-bank)
- **B3** MEMIT *labeled* "weight-editing baseline" (LLM weights modified — for comparison only, not our method)
- **B4** LoRA *labeled* "fine-tune baseline" (LLM weights modified — for comparison only)

We will not claim "Mneme beats MEMIT" or "Mneme beats LoRA" because B3/B4 modify the LLM. The comparison frame is: **"what can Mneme deliver while leaving the LLM frozen?"**

## 5. Hegel-prompt qualitative (planned, Phase N3)

The user's reference prompt about Hegel's `negation of negation` will be run on each ready model with and without a small "philosophy primer" written into the bank, with side-by-side outputs in `transcripts/hegel/<model>/`. This is a smell-test, not statistical evidence.

## 6. What we do not (yet) claim

- Mneme generalizes across **languages**. We have only English. Bilingual eval is roadmap (`v3.2`).
- Mneme survives **multi-hop reasoning**. Stage 14 was 1-hop facts; Phase G held-out was paraphrase-only.
- Mneme beats RAG at **scale > 10k facts**. The bank is per-layer dense; storage scales linearly.
- Mneme works with **Flash-Attention**. We patch the eager forward only.

These are roadmap items for v3.2+ and listed honestly so reviewers don't have to guess.
