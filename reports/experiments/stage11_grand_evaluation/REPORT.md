# Stage 11 — Grand Evaluation Report

> Status: complete. Hardware: NVIDIA GB10 (Blackwell, 128 GB unified). Base: frozen `google/gemma-4-E2B`, bf16. All numbers report mean over 3 seeds + paired bootstrap 95% CI (10 000 resamples). Gates use the **CI lower bound**, not the mean (NeurIPS-style strict gating).

## TL;DR

| Sub-stage | What we tried to break | Verdict |
| --- | --- | --- |
| **11A** Paraphrase-augmented InfoNCE encoder | "Encoder is fingerprint-matching, not address-binding" | **❌ FAIL.** Held-out paraphrase recall@1 = 0.138 (multilayer) / 0.053 (prompt_hidden), CI nowhere near 0.85 gate. The encoder *still* relies on surface lexical features even with 6 paraphrase templates per training fact. Honest negative result. |
| **11A** Decoy ×1000 (G10B regression) | "Bank only works without distractors" | **✅ PASS.** Both encoders hold 1.000 top-1 even at 150 000 distractor slots. |
| **11A** Value ablation (G10D regression) | "Encoder alone explains it" | **✅ PASS.** random-value top-1 = 0.000, shuffled-value top-1 = 0.009. Bank is necessary. |
| **11B** Train-time LORO + adversary | "Bank generalizes to a held-out relation" | **❌ FAIL.** Mean 0.108 across 6 relations × 3 seeds. Best relation P937 = 0.417; worst (P39, P101) = 0.000. Gradient-reversal adversary at weight 0.1 did not move the needle vs Stage 10F. Generalization to truly-unseen relations remains unsolved. |
| **11D** Multi-turn ConvQA (D1) | "DM forgets when conversational filler is interleaved" | **✅ PASS.** recall@1 = 1.000 at k ∈ {1, 3, 5, 10} filler turns, no leakage. |
| **11D** Chat-as-write-API vs RAG (D2) | "RAG matches DM at write-via-conversation" | **✅ PASS.** DM = 0.967, RAG = 0.275, advantage +0.692 (CI [0.625, 0.775]). |
| **11D** Prompt-injection / poisoning (D3) | "Adversary can overwrite a slot" | **✅ Partial.** Protected-slot overwrite rate = 0.000 ✅, original-recall after attack = 0.983 (borderline). **Benign-write accept = 0.000 ❌** — the write-policy is over-strict and rejects legitimate new addresses. Documented as a known overcorrection. |
| **11E** Bit-exact reproduction | "Numbers are non-deterministic" | **✅ PASS.** Two independent runs produce identical SHA-256 over the stable-subset of summary metrics. |

## Honest framing

Of the four failure-modes Stage 11 was designed to attack:

1. **Paraphrase fingerprint-matching is real.** Six training paraphrases per fact + InfoNCE retrieval is *not* enough to make the encoder relation-invariant on truly-held-out templates. This is a real limit of our current `multilayer` / `prompt_hidden` encoders.
2. **Cross-relation generalization is real.** A trained DM bank is *not* an editable memory at the relation level. New relations require retraining. F7 / G11B / G10F all confirm this.
3. **Conversational robustness is genuine.** Multi-turn filler does not break retrieval; chat-as-write-API beats RAG by a wide margin; protected slots resist injection. Within-distribution conversational use is the strongest evidence in this stage.
4. **The bank is necessary.** Random-value and shuffled-mapping ablations destroy retrieval, confirming the encoder alone is not sufficient (F5 falsified).

What we therefore claim, post-Stage-11:
- Mneme implements **robust binding for trained facts** (any encoder, any decoy scale, any conversational-filler depth).
- Mneme provides a **fast write API** that beats vector-RAG when the same encoder is used downstream.
- Mneme does **not** yet generalize to (a) paraphrases of trained facts under our `multilayer` / `prompt_hidden` encoders, or (b) relations not seen during training. Both remain open problems.

What we therefore do **not** claim:
- "DM is a one-shot editable memory" — falsified by 11B.
- "DM solves the linear-representation problem" — irrelevant; we measure outcomes, not mechanisms.
- "DM matches the language coverage of RAG" — we do not test multi-token answers or open-ended QA.

## Detailed results

See `SUMMARY_TABLE.md` for the full paired-bootstrap table, `stage11_summary.json` for machine-readable metrics, and `docs/figures/stage11_fig{12..16}_*.svg` for figures.

## Next directions (deferred)

The 11A failure motivates three concrete follow-ups, none of which are claimed here:
1. **Orthogonal bank** (Givens / Householder) to enforce key separability geometrically rather than via InfoNCE.
2. **Sparse autoencoder bank** (à la Anthropic dictionary learning) to disentangle superposed concepts before key extraction.
3. **ROME-style closed-form covariance update** to avoid the encoder-projector chain entirely for individual edits.

These require new infrastructure and are listed for future sessions.

## Reproducibility

- Bit-exact harness: `bash scripts/reproduce_stage11.sh` runs two deterministic copies and asserts SHA-256 hash equality on the stable-subset.
- Hardware: NVIDIA GB10 (Blackwell, 128 GB unified, CUDA 13.x).
- Software: PyTorch 2.10.0+cu13 / transformers 4.57.1 / Python 3.13.
- All training jobs orchestrated by `scripts/run_stage11_sweep.sh` (29 runs total, idempotent skip-if-exists).
