# V2 Final Verdict — Hippocampus-style Native LLM Memory (HNM)

> **Document Status**: DRAFT — Sections marked `[TBD:eXX]` will be filled as experiments e01-e19 complete. This verdict will be sealed when all experiment verdicts are written, V1_CLOSEOUT is complete, and user signs off.

---

## 1. Abstract

**Central Claim Restated**: A frozen LLM, augmented with (i) a per-position learnable pause head, (ii) a per-layer K/V projector over a shared AttentionBank, and (iii) a multi-round halt mechanism, can integrate long-term hippocampus-style memory and within-inference working memory through native attention—no fine-tuning of base weights, no prompt rewriting, no external retriever calls.

**Strongest Evidence to Date**:
- **Phase B2 (v1/Exp42)**: On Qwen3-4B-Instruct-2507 (frozen), preloading 512 MEMIT b-vectors into layer-9 AttentionBank + training a rank-64 (I+P) residual K/V projector (420K params, 200 steps) achieved test NLL **12.13 → 6.30 (Δ=−5.83)**. Control with random bank + same trained projector yielded Δ=−0.02, confirming the improvement is specific to preloaded content, not architectural bias.
- **E01 (v2, partial, seed 0)**: Four critical falsifiers PASS at Δ=−3.90 (v2 canonical reproduce):
  - **H1 (bank-off)**: Trained projector with bank removed at eval returns NLL to base (11.998), proving improvement requires bank content.
  - **H4 (zero bank)**: Replacing bank entries with zeros at eval also returns NLL to base, proving non-zero structure is necessary.
  - **H7 (random-bank train)**: Training same projector from step 0 with random bank achieves only Δ=−0.29 (gap=3.61 vs real bank), proving projector specifically reads preloaded content.
  - **H2 (row-shuffle, per-row permutation)**: Unexpectedly did NOT kill the effect (Δ=−4.62), reframing the claim: the mechanism provides **addressable slots** via (bank-entry identity, learned projector) joint—not via b-vector internal geometry alone. This is analogous to RAG: retrieved chunks need not be in model's native space; the projector places them there.

**Limit Cases Discovered**:
- **H2 reframe required**: Per-row dimension shuffle preserves improvement, indicating information lives in per-entry addressability, not in individual b-vector's coordinate structure. A stricter test (same permutation across all rows, destroying per-row identity) is pending to confirm the reframed hypothesis.
- **Single-domain evidence**: Current evidence is limited to factual completion (MEMIT-style triplet queries). No multi-task generalization (GSM8K, StrategyQA, etc.) has been tested yet.
- **No end-to-end multi-round K>1 proof**: Phase B2 and E01 use K=2 rounds, but halt mechanism, ACT-style dynamic pondering, and K curriculum have not been validated.
- **Pause-head training not demonstrated in v2**: Phase B2 did not train pause heads; auto-pause mechanism remains unvalidated at v2 scale.

**Overall Stance**: **Substantially falsified in its original framing**; see §1b (Headline finding, revised). The mechanism is real but its interpretation has changed. The K-projector + non-empty bank produces a reliable NLL drop, but e11 wave-3 falsifies the "memory content matters" reading: random Gaussian banks, single-row-replicated banks, and constant-vector banks all yield NLL drops in the same range as (or larger than) the canonical real bank. The remaining defensible claim is narrower: the v2 architecture is a **parameter-efficient adapter** for a frozen LM. Cross-model (e05), capability drift (e03), and multi-round (e04, e14, e15) experiments are still required, but the *headline framing* of the project must be revised before any of those run.

---

## 1b. Headline finding (revised)

**The original v2 thesis — "memory content matters, the bank stores facts that the projector reads out" — is substantially falsified by e11 wave-3 (seed 0).**

What the e11 wave-3 ablations show (all seed 0, same projector/training/eval pipeline as e01 canonical; raw JSONs in `v2/experiments/e11_noise_robustness/`):

| Bank construction at preload | Row distinctness | Δ NLL (real eval) | Post-train `off=base`? |
|---|---:|---:|---|
| canonical real b-vectors (reference) | 18.70 | **−3.90** | ✅ 11.998 |
| **n1**: iid Gaussian, renormed L2=15 (pure noise) | 21.21 | **−6.05** | ✅ 11.998 |
| **n3**: ONE real row, replicated 512× | 0.00 | **−5.50** | ✅ 11.998 |
| **n5**: a constant vector, replicated 512× | 0.00 | **−2.83** | ✅ 11.998 |
| **n7**: K=0 pure projector (empty bank) | — | crashed at backward() | n/a — cannot train P without bank tokens |

**Wave-5 replication at L21 (deeper layer):** the same falsification holds at L21, where the *real* bank's strongest layer also lives.

| Bank construction at preload (L21) | Δ NLL | vs real-bank L21 reference |
|---|---:|---|
| canonical real b-vectors (L21, seeds 0/1/2) | −6.35 / −6.53 / −6.52 | reference |
| **n1**: iid Gaussian L2=15 at L21 | **−6.32** | matches real bank |
| **n3**: one real row × 512 at L21 | **−6.36** | matches real bank |
| **n5**: constant vector × 512 at L21 | **−6.48** | matches real bank |

The capacity reading therefore holds at *both* the layer where v1's claim lived (L9) and the layer where the strongest v2 signal lives (L21). The "L21 wins because deeper memory reads richer features" interpretation is also dead — L21 wins because that's where a rank-64 residual adapter has the most headroom.

Two facts emerge cleanly:

1. **Bank substrate is necessary.** In every variant, emptying the bank at eval returns NLL to the base 11.998 exactly. Take the bank away → the trained projector cannot act. n7 confirms the dual: with no bank tokens at training time, the bank-side projector has no gradient path and the training crashes. **The non-empty bank is a structural requirement, not a content store.**
2. **Bank content is NOT what is being read.** Random Gaussian noise gives a *bigger* NLL drop than the real bank. A single row replicated 512 times (zero distinctness) still gives −5.50. A constant vector — every slot identical — still gives −2.83. None of these banks contain factual information of any kind, and yet the same projector training produces the same (or stronger) Δ NLL.

The original interpretation — "preloaded b-vectors hold compressed knowledge that the projector decodes" — is incompatible with these results. It is dead.

**The actual mechanism, restated.** A rank-64 trainable layer-9 K/V projector, given **any** non-empty bank as attention substrate, adapts the frozen LM toward the train distribution. The bank provides extra K/V positions for the projector to write into; its content is a substrate variable, not an information carrier. This is functionally equivalent to a tiny LoRA-style adapter that happens to be plumbed through an attention-bank API. The H2c result (collapsed bank, all rows = mean, Δ=−4.81) was the first warning; e11 wave-3 confirms it across the noise / single-row / constant-vector axes.

**Supporting evidence consistent with the "adapter, not memory" view:**
- **Layer sweep (seed 0)**: L3=−1.58, L9=−3.90, L21=**−6.29** (best), L33=−3.97. The improvement scales with where in the residual stream a small additive transform is most useful — a property of adapter placement, not of memory geometry.
- **H3 (subject+relation disjoint OOD, seed 0)**: Δ=−2.69 PASS. The projector generalizes beyond memorized routing, which is *consistent* with a content-blind adapter (the adapter helps anywhere the train distribution is informative about the eval distribution), not specifically with retrieval.
- **e03 capability drift (seed 0, WikiText-2)**: base ppl=8.349, bank_on ppl=8.380 (relative drift +0.37%, well under the 5% threshold). The adapter does not corrupt general LM capability — exactly what one expects from a small low-rank residual.

**Implications for the project.**

1. **Framing**: v2 must be presented as a **parameter-efficient adapter that uses an AttentionBank API as its surface**, not as a hippocampus-style memory system. Any text in v2 documents implying "facts are stored in the bank and retrieved" must be removed or qualified.
2. **What v1's Phase B2 −5.83 actually was**: a real NLL improvement from a real low-rank adaptation, mislabeled as memory retrieval. The number reproduces; the *interpretation* was wrong.
3. **The hippocampus analogy is currently unsupported.** No part of e01 + e11 wave-3 shows content-conditional retrieval. Pause-write working memory (e14) and dual-channel interrupt (e11-dual, e08) remain untested and cannot rescue this claim on their own — they are *additional* mechanisms, not replacements for the missing content-sensitivity in the core read path.

**Three open paths forward** — these are the only experiments that can either rehabilitate the original thesis or sharpen the revised one:

- **(a) e10 — top-K retrieval.** All e01 / e11 evidence used *all-attend* over the bank. If retrieval is content-sensitive, restricting attention to the top-K most similar slots should (i) help on real banks and (ii) collapse on random/collapsed banks. Top-K is the *only* mechanism that can recover a content-sensitivity signal from the current architecture. If e10 top-K with real bank still matches all-attend Δ, retrieval is dead.
- **(b) e13 — cross-task transfer.** Under the "capacity / adapter" reading, the projector trained on factual completion should help on *unrelated* tasks too, because what it learned is a generic residual adjustment to layer-9. Under the (now-falsified) "content read" reading, it should not. e13 is the cleanest disambiguator.
- **(c) e06 — relation-disjoint OOD.** H3 already passed for subj+rel disjoint at the data-split level. e06 with strict relation-disjoint splits will distinguish "adapter generalizes broadly" (Δ still negative) from "adapter only helps where train and test share relation structure" (Δ→0). Capacity story predicts the first; retrieval story predicts the second.

**Bottom line.** The v2 mechanism produces a real, reproducible, capability-preserving NLL improvement using ~602K trainable parameters over a frozen 4B model. That part of v1's Phase B2 stands. Everything else — the AttentionBank framing, the LT/ST memory dichotomy, the hippocampus analogy — is unsupported by the falsifiers run to date and, on the most rigorous reading, should be retracted pending e10 / e13 / e06.

---

## 2. Headline Numbers

| Experiment | Metric | Value | Status |
|---|---|---:|---|
| **e01-canonical** | Δ NLL (base → B2-trained) | −3.90 | ✅ PASS (seed 0, N=120) |
| **e01-h1** | NLL post bank-off | 11.998 (≈ base) | ✅ PASS (H1 falsified) |
| **e01-h2** | Δ NLL after row-shuffle | −4.62 | ⚠️ REFRAMED (see §3) |
| **e01-h3** | Δ NLL entity+relation disjoint | −2.69 | ✅ PASS (consistent w/ adapter; not diagnostic of retrieval) |
| **e01-h4** | NLL with zero bank | 12.00 (≈ base) | ✅ PASS (H4 falsified) |
| **e01-h5** | N_preload sweep monotonicity | [TBD:e01] | pending |
| **e01-h6** | Layer sweep (≥3 layers Δ ≤ −1.0) | L3=−1.58, L9=−3.90, L21=**−6.29**, L33=−3.97 | ✅ PASS (4/4 ≤ −1.0; deeper > shallower) |
| **e01-h7** | Δ NLL random-bank train | −0.29 (gap=3.61) | ⚠️ SUPERSEDED by e11/n1 (random L2=15 → Δ=−6.05) |
| **e01-h8** | Logit KL on neutral sentences | [TBD:e01] | pending |
| **e01-h9** | Cross-model smoke (Qwen3-1.7B) | [TBD:e01] | pending |
| **e01-h10** | Gate selectivity entropy | [TBD:e01] | pending |
| **e01-h11 (e11/n1)** | Δ NLL on pure random L2=15 bank | **−6.05** | ❌ FALSIFIES content-read claim |
| **e01-h11 (e11/n3)** | Δ NLL on single-row replicated bank | **−5.50** | ❌ FALSIFIES content-read claim |
| **e01-h11 (e11/n5)** | Δ NLL on constant-vector bank | **−2.83** | ❌ FALSIFIES content-read claim |
| **e02** | Scale matrix (N_preload, N_train, lr, steps) | [TBD:e02] | not started |
| **e03** | WikiText-2 PPL drift (rel%) | base=8.349 / bank_on=8.380 → **+0.37%** | ✅ PASS (≪ 5% threshold) |
| **e03** | lm-eval acc drop (pp) | [TBD:e03] | not started |
| **e04** | ACT halt mean K_used | [TBD:e04] | not started |
| **e04** | Spearman(K_used, NLL_drop) | [TBD:e04] | not started |
| **e05** | Qwen3-1.7B Δ NLL | [TBD:e05] | not started |
| **e05** | Qwen3-4B Δ NLL (5k steps) | [TBD:e05] | not started |
| **e05** | Llama-3.2-3B Δ NLL | [TBD:e05] | not started |
| **e05** | Mistral-7B Δ NLL | [TBD:e05] | not started |
| **e06** | Relation-disjoint OOD Δ NLL | [TBD:e06] | not started |
| **e07** | Multi-layer vs single-layer gain | [TBD:e07] | not started |
| **e08** | Interrupt API demo (GSM8K acc boost) | [TBD:e08] | not started |
| **e09** | v1 AttnNativeBank + KProj gate_d boost | [TBD:e09] | not started |
| **e10** | topK=16 perf vs all-attend (%) | [TBD:e10] | not started |
| **e10** | topK=16 compute vs all-attend (%) | [TBD:e10] | not started |
| **e11** | Dual-channel (auto+interrupt) gain vs single | [TBD:e11] | not started |
| **e12** | LT+ST vs max(LT, ST) gain | [TBD:e12] | not started |
| **e13** | Multi-task (≥3 tasks +2pp vs base) | [TBD:e13] | not started |
| **e14** | Pause head mean rate | [TBD:e14] | not started |
| **e14** | Pause head NLL drop vs B2 (%) | [TBD:e14] | not started |
| **e15** | K curriculum final Δ NLL | [TBD:e15] | not started |
| **e16** | Optimal capacity C (NLL vs latency) | [TBD:e16] | not started |
| **e17** | Negation robustness Δ NLL | [TBD:e17] | not started |
| **e18** | 2-hop chaining Δ NLL | [TBD:e18] | not started |
| **e19** | Seed replication std/mean ratio | [TBD:e19] | not started |

---

## 3. H1-H15 Falsifier Scoreboard

From V2_METHODOLOGY_DEBATE.md §1, the 15 cheap explanations that must be falsified:

| H | Cheap Explanation | Falsification Method | Pass Criteria | Result (seed 0) | Result (seed×3) |
|---|---|---|---|---|---|
| **H1** | Projector learns vocab bias (any b → high-freq token) | e01-1: eval with bank-off (frozen trained projector, empty bank) | NLL ≥ base − 0.05 | ✅ **PASS** (11.998 ≈ base) | [TBD:e01] |
| **H2** | b-vectors encode target embeddings; projector is decoder shortcut | e01-2: row-level shuffle b dimensions (per-row perm) | NLL degrades ≥ +4.0 vs canonical | ❌ **FAILED TO FALSIFY** (−4.62, no degrade) — see e11 wave-3 below | [TBD:e01] |
| **H2b** | (revised test) All rows same permutation → destroy per-row identity | e01-2b: apply identical permutation to all rows | NLL degrades ≥ +4.0 vs canonical | superseded by e11/n3,n5 (zero-distinctness banks still help) | [TBD:e01] |
| **H3** | Train/test entity or relation leakage | e01-3: strict entity-disjoint + relation-disjoint splits | Δ NLL still ≤ −1.0 | ✅ **PASS** (Δ=−2.69) — *but not diagnostic of retrieval; an adapter also generalizes* | [TBD:e01] |
| **H4** | Random-bank control is buggy (randomization bypassed) | e01-4: zero bank.slots directly (explicit null) | NLL ≈ base ±0.02 | ✅ **PASS** (12.00 ≈ base) | [TBD:e01] |
| **H5** | One bank entry enough (pseudo-retrieval, size-insensitive) | e01-5: N_preload sweep {1,2,4,16,64,256,512,2048,8192}×seed×3 | NLL monotonically improves with N | [TBD:e01] | [TBD:e01] |
| **H6** | Layer-9 selection accidental (any layer would work) | e01-6: layer sweep {3,9,15,21,27,33}×seed×3 | ≥3 layers achieve Δ ≤ −1.0 | ✅ **PASS** L3=−1.58, L9=−3.90, L21=−6.29, L33=−3.97 (4/4) | [TBD:e01] |
| **H7** | Random bank + projector training achieves same performance | e01-7: train random-bank from scratch 200 steps | random stays ≥ 4.0 NLL above real bank | ⚠️ **SUPERSEDED**: at L2=1, Δ=−0.29 (gap=3.61); at correct L2=15 (e11/n1), Δ=−6.05 — random *beats* real | [TBD:e01] |
| **H8** | General LM capability destroyed (overfitting to fact-recall) | e01-8 + e03: WikiText-2/103 PPL + lm-eval (HellaSwag, ARC, MMLU) | rel PPL drift ≤ 5%, acc drop ≤ 2pp | ✅ **PASS** (WikiText-2: 8.349→8.380, +0.37%); lm-eval [TBD:e03] | [TBD:e01, e03] |
| **H9** | Single-model fluke (Qwen3-4B architecture quirk) | e01-9 + e05: cross-model (Qwen3-1.7B, Llama-3.2-3B, Mistral-7B) | ≥1 model achieves Δ ≤ −1.0 | [TBD:e01, e05] | [TBD:e01, e05] |
| **H10** | Bank_gate learns degenerate on/off (always 0 or 1) | e01-10: gate histogram + per-layer entropy (H = −Σ p log p) | gate selectivity entropy > 0.3 nats | [TBD:e01] | [TBD:e01] |
| **H11** | K-projector rank-64 over-parameterized (memorizing test set) | e02: projector rank sweep {8,16,32,64,128} + N_train {120,1k,5k} | lower rank (16/32) still Δ ≤ −2.0 | [TBD:e02] | [TBD:e02] |
| **H12** | 200 steps cherry-picked convergence point | e02: step sweep {50,100,200,500,1000,5000} | improvement plateaus after 200, no U-shape | [TBD:e02] | [TBD:e02] |
| **H13** | Test set accidentally similar to train (correlation) | e06: relation-disjoint OOD (train ∩ test = ∅ on relations) | Δ NLL ≤ −0.5 on OOD | [TBD:e06] | [TBD:e06] |
| **H14** | Effect tied to specific prompt format or tokenization | e13: multi-task eval (GSM8K, StrategyQA, NegQA, CSQA) | ≥2 tasks show Δ ≤ −0.5 | [TBD:e13] | [TBD:e13] |
| **H15** | Learning rate 2e-4 tuned to this exact data | e02: lr sweep {5e-5,1e-4,2e-4,5e-4,1e-3} | ≥2 lr values achieve Δ ≤ −3.0 | [TBD:e02] | [TBD:e02] |
| **H_content** (e11 wave-3 + wave-5) | Bank content carries the information that is read out | e11: random Gaussian / single-row / constant-vector banks at L9 **and** L21 | random/degenerate banks should give Δ ≈ 0 at every layer | ❌ **FALSIFIED at L9 and L21**: L9 n1=−6.05, n3=−5.50, n5=−2.83; **L21 n1=−6.32, n3=−6.36, n5=−6.48** (all four bank constructions match real-bank L21 reference −6.35/−6.53/−6.52) | [TBD] |

**Current Falsifier Pass Rate**: At seed 0: H1 ✅, H3 ✅, H4 ✅, H6 ✅, H8 ✅ (WikiText-2 portion); H2 ❌ failed to falsify (content claim survives the test but is killed by e11); H7 ⚠️ superseded; **H_content ❌ definitively falsifies the content-read interpretation**. Pre-e11 the scoreboard read "5/15 PASS, on track"; post-e11 the scoreboard reads "5/15 PASS at the *adapter* claim, 0/1 PASS at the *content-read* claim — original thesis is dead." See §1b.

**Revised pass criteria.** The "≥12/15 → claim supported" rule was written for the original "memory content matters" thesis. Because e11 falsifies that thesis directly via H_content, the H1-H15 pass count is no longer the binding criterion. The binding criteria going forward are:
1. e10 top-K must show **content-sensitivity asymmetry** (real bank > random/collapsed bank by ≥ 2.0 NLL under top-K) — otherwise the retrieval channel is dead and v2 is purely an adapter.
2. e13 cross-task must show **transfer ≥ +0.5 pp on ≥2 unrelated tasks** — this confirms the adapter reading and gives v2 a defensible, narrower headline.
3. e06 relation-disjoint must show **Δ ≤ −1.0** on strict relation OOD — adapter prediction; if it fails, even the adapter claim is task-specific.

If (1) succeeds, the memory framing partially recovers. If (1) fails and (2)+(3) succeed, v2 is reframed and shipped as an adapter. If all three fail, v2 has no defensible headline beyond "reproduces v1's −3.90 number under tighter falsifiers."

**Why H2 + H3 are no longer evidence for the memory thesis.** H2's failure to degrade and H3's OOD pass were originally read as "the projector learns a non-trivial read mechanism." e11 reframes both: an adapter would also pass H3 (training-distribution adaptation transfers to held-out subject+relation pairs) and would not be touched by per-row shuffles (the shuffle preserves per-row L2 and rough distribution). Neither result distinguishes adapter from retrieval. Only e10 top-K can.

---

## 4. Cross-Model Evidence

**Purpose**: Establish that the K-projector mechanism generalizes across LLM architectures, ruling out Qwen3-4B-specific artifacts (H9).

**Planned experiments (e05)**:
- Qwen3-1.7B-Instruct: Full anti-cheat suite (H1-H10) at 1k steps, N_preload=512, N_train=1000, N_test=300
- Qwen3-4B-Instruct-2507: Extended training (5k steps) to test convergence plateau (H12)
- Llama-3.2-3B-Instruct: Cross-family replication at 1k steps
- Mistral-7B-Instruct-v0.3: Larger model (if MPS 128GB permits) at 2k steps

**Pass criteria**: ≥2 models achieve Δ NLL ≤ −1.0 on held-out test set (disjoint from train).

**Results**: [TBD:e05]

---

## 5. Capability Drift

**Purpose**: Ensure that training the K-projector + bank_gate heads does not degrade the frozen base model's general language modeling capabilities (H8).

**Planned experiments (e03)**:
- **WikiText-103 validation set**: ≥100K tokens, 4 configs:
  1. Base model (no projector, no bank)
  2. B2-trained projector + bank-off (projector loaded, bank disabled at eval)
  3. B2-trained projector + bank-on (canonical B2 setup)
  4. B2-trained projector + interrupt-on (with interrupt API enabled, oracle hints injected)
- **lm-eval-harness subset**: HellaSwag, ARC-easy, MMLU-stem (100 questions each)

**Pass criteria**:
- Relative WikiText-103 PPL drift ≤ 5% (i.e., PPL_trained / PPL_base ≤ 1.05)
- lm-eval accuracy drop ≤ 2 percentage points across all three tasks

**Results (seed 0, WikiText-2 partial)**:
- WikiText-2 base ppl = **8.349**, B2-trained projector + bank-on ppl = **8.380**
- Relative drift = **+0.37%** — well below the 5% threshold → ✅ **PASS** on the WikiText axis
- lm-eval-harness (HellaSwag, ARC-easy, MMLU-stem): [TBD:e03]
- WikiText-103 full validation set: [TBD:e03] (only WikiText-2 run so far)

**Reading**: The adapter does not measurably degrade general LM capability — exactly what one expects from a 602K-param low-rank residual at a single layer. This is one of the few results that *increases* confidence in the revised (adapter) framing in §1b: a true capability-destroying memory injection would surface here, and it does not.

**Remaining drift work**: [TBD:e03]

**Rationale**: Training 602K params (0.015% of base model) should not affect frozen weights. If drift exceeds thresholds, it suggests the projector/gate training introduces numerical instabilities or optimization artifacts that indirectly corrupt activations. This would trigger abandonment condition 4.1.3 (see V2_METHODOLOGY_DEBATE.md).

---

## 6. What HNM Uniquely Delivers

From V2_DIFFERENTIATION.md §8, the five core capabilities that distinguish HNM from all alternative approaches (RAG, LoRA, ICL, CoT, MEMIT):

1. **Latent memory injection**: Hidden-state vectors (h ∈ ℝ^d) enter the model at target layers via attention K/V, bypassing tokenization entirely. This enables injection of non-verbalizable internal states (e.g., mid-computation pause-writes, oracle hints from external debuggers).

2. **Frozen base model**: All Transformer weights (W_Q, W_K, W_V, W_O, MLP, embeddings) remain untouched. Only a small projector (rank-r, ~460K params for Qwen3-4B) and gate heads (~140K params) are trained. Storage per task: 1-5 MB vs LoRA's 10-100 MB. Trainable params: 0.015% of base model vs LoRA's ~2.75%.

3. **Scalable external memory**: Bank supports N_b ∈ {1, …, 10K+} entries without degrading the model (unlike MEMIT, which collapses after ~1000 edits). Memory is explicitly stored as h-vectors, not implicitly in weights—allowing inspection, editing, and injection at runtime.

4. **Dual-channel write**: (i) **Auto-pause**: Learnable pause heads trigger mid-layer skips, writing hidden states to bank for next-round retrieval (working memory). (ii) **Human interrupt**: Public API `interrupt(model, round, layer, pos, h)` allows external programs to inject latent context at arbitrary (round, layer, position) coordinates—no prompt rewriting required.

5. **Multi-round pondering without token generation**: K_max rounds of attention (K ∈ {2,4,8}) with ACT-style halt enable the model to "think" iteratively by reading/writing to the bank across rounds. This incurs K × (forward pass overhead) but generates **zero extra output tokens**, unlike CoT which generates 10-100× more tokens for reasoning steps.

**Bottom line**: HNM is not RAG (no text retrieval), not LoRA (frozen weights), not ICL (bank, not prompt), not CoT (rounds, not tokens), not MEMIT (external bank, not weight edits). It is a **native attention-based memory mechanism** for frozen LLMs that integrates long-term (preloaded) and short-term (pause-write) memory through learnable projectors and multi-round inference.

---

## 7. Limitations and Known Failure Modes

**Brutally honest assessment** of current evidence gaps and known weaknesses:

1. **Single-domain evidence only**: All experiments to date (Phase B2, E01) use factual completion tasks (MEMIT-style triplet queries: "Paris is the capital of ___"). No evidence yet for: mathematical reasoning (GSM8K), commonsense QA (StrategyQA), multi-hop reasoning (HotpotQA), negation robustness (NegQA), or cross-task generalization. **Risk**: The mechanism may be specific to entity-relation-target triplets and fail on open-ended reasoning.

2. **H2 reframe reveals weaker claim**: The per-row dimension shuffle (H2) did **not** kill the effect, forcing a reframe: the b-vector's internal geometry does not carry the answer; instead, the mechanism provides **addressable slots** via per-entry identity + learned projector. This is less impressive than "b-vectors encode compressed knowledge"—it's closer to "projector learns a lookup table into frozen QK space." The revised test (H2b: same permutation across all rows) may yet falsify the claim if per-row addressability is also unnecessary (e.g., if only L2 norm matters).

3. **No end-to-end multi-round inference at K>1 demonstrated**: Phase B2 and E01 use K=2 rounds (one standard forward, one bank-injected forward). The ACT halt mechanism (e04), K curriculum (e15), and ponder loss training have not been validated. **Risk**: Multi-round dynamics may not converge, or the halt head may learn degenerate behavior (always halt at K=1, or never halt). If e04 fails (mean K_used ≈ 1.0 or ≈ K_max with no task-dependent modulation), the "multi-round pondering" claim collapses.

4. **Interrupt API implemented but not stress-tested**: The `interrupt(model, round, layer, pos, h)` public API exists in `v2/core/interrupt_api.py`, but only a qualitative demo (e08) is planned. No systematic study of: injection timing sensitivity, injection layer sensitivity, injection volume (how many positions can be injected before bank capacity saturates), or injection content quality (oracle vs noisy hints). **Risk**: The interrupt mechanism may be fragile—working only for oracle hints at precisely tuned (layer, round) coordinates.

5. **Pause-head training not shown to converge in v2**: Phase B2 froze the pause heads (no pause-write). Experiment e14 will attempt to train pause heads with entropy regularization, but convergence is uncertain. Historical v1 experiments (Exp38 LPL Phase A) showed pause heads collapsing to always-on or always-off without careful curriculum + λ_pause tuning. **Risk**: Auto-pause mechanism may be untrainable at scale, leaving HNM as "static LT memory only" (no working memory component).

6. **No cross-task generalization study yet**: Experiment e13 will test multi-task capability (GSM8K + StrategyQA + NegQA + CSQA + simple-QA mixed batch), but this is unvalidated. **Risk**: The projector may overfit to factual completion and fail to generalize. If <3 tasks show improvement ≥ +2pp vs base, the claim must be downgraded to "single-task HNM."

7. **Capability drift unchecked**: WikiText-103 PPL and lm-eval-harness (e03) have not been run. **Risk**: Training the projector may introduce numerical artifacts (e.g., gradient noise from untrained frozen layers, quantization drift in bf16 MPS) that degrade general LM capability. If PPL drift > 10% or acc drop > 5pp, the mechanism is **unacceptable** (abandonment condition 4.1.3).

8. **Layer-9 selection is pre-chosen from B2**: E01-h6 will sweep layers {3,9,15,21,27,33}, but if only layer 9 works and all others fail (Δ > −1.0), this suggests the effect is tied to a specific layer's geometry in Qwen3-4B, not a general principle. This would weaken cross-model generalization (e05) and limit practical deployment (users would need layer-search for each model).

9. **Train/test split not yet entity+relation disjoint**: E01 canonical and H2/H4/H7 use the default disjoint split from `v2/core/data_io.disjoint_split`, which ensures entity-disjoint but may allow relation overlap. E01-h3 will enforce strict relation-disjoint (train relations ∩ test relations = ∅). If this kills the effect (Δ > −1.0), it suggests the projector is learning relation-level shortcuts, not entity-level addressability.

10. **Random-bank control gap (3.61 NLL) is large but not infinite**: H7 shows random-bank training achieves Δ=−0.29 (vs real-bank Δ=−3.90, gap=3.61). This is strong evidence the projector reads preloaded content, but the gap is not as extreme as "random → zero improvement." **Risk**: The random bank may provide a weak regularization benefit (e.g., increasing effective key-space dimensionality), and the real bank's benefit may be partially attributable to this generic expansion, not to semantic content. A deeper control (e.g., random bank with **same L2 norms** as real bank) may reduce the gap.

---

## 8. Reproduction Summary

**One-liner per experiment with exact CLI to reproduce.** All commands assume `cwd=/Users/gabiri/projects/RCV-HC` and use `--device mps` for Apple Silicon. For seed sweeps, repeat with `--seed 1` and `--seed 2`.

### Phase B2 (v1, original)
```bash
python3 v1/experiments/atb_validation_v1/exp42_lpl/06_phase_b2_kproj.py \
    --device mps --steps 200 --lr 2e-4 --rank 64
```

### E01: Anti-cheat suite (falsifiers H1-H10)
```bash
# Canonical (B2 reproduce)
python3 v2/experiments/e01_anticheat_b2/run.py --variant canonical --seed 0 --n_test 120

# H1: Bank-off ablation (post-training eval with empty bank)
python3 v2/experiments/e01_anticheat_b2/run.py --variant h1_bank_off --seed 0 --n_test 120

# H2: Row-level shuffle b dimensions (per-row permutation)
python3 v2/experiments/e01_anticheat_b2/run.py --variant h2_shuffle_b --seed 0 --n_test 120

# H3: Entity+relation disjoint split
python3 v2/experiments/e01_anticheat_b2/run.py --variant h3_disjoint_split --seed 0 --n_test 120

# H4: Zero bank slots (replace with zeros at eval)
python3 v2/experiments/e01_anticheat_b2/run.py --variant h4_zero_bank --seed 0 --n_test 120

# H5: N_preload sweep {1,2,4,16,64,256,512,2048,8192}
for N in 1 2 4 16 64 256 512 2048 8192; do
    python3 v2/experiments/e01_anticheat_b2/run.py --variant h5_n_sweep --n_preload $N --seed 0 --n_test 120
done

# H6: Layer sweep {3,9,15,21,27,33}
for L in 3 9 15 21 27 33; do
    python3 v2/experiments/e01_anticheat_b2/run.py --variant h6_layer_sweep --bank_layer $L --seed 0 --n_test 120
done

# H7: Train random bank from step 0
python3 v2/experiments/e01_anticheat_b2/run.py --variant h7_rand_train --seed 0 --n_test 120

# H8: Logit KL on neutral sentences (post-train)
python3 v2/experiments/e01_anticheat_b2/run.py --variant h8_kl_neutral --seed 0 --n_neutral 1000

# H9: Cross-model smoke (Qwen3-1.7B, quick sanity)
python3 v2/experiments/e01_anticheat_b2/run.py --variant h9_cross_smoke --model Qwen/Qwen3-1.7B-Instruct --seed 0 --steps 60 --n_test 120

# H10: Gate histogram + per-layer entropy
python3 v2/experiments/e01_anticheat_b2/run.py --variant h10_gate_hist --seed 0 --n_test 120
```

### E02: Scale matrix (N_preload × N_train × layers × lr × steps)
```bash
# [TBD:e02] — Latin-square subset of full grid, ~60 configs
# Example single config:
python3 v2/experiments/e02_scale_matrix/run.py \
    --n_preload 2048 --n_train 1000 --n_test 300 \
    --layers multi_9_15_21 --lr 2e-4 --steps 1000 --seed 0
```

### E03: Capability drift (WikiText-103 + lm-eval-harness)
```bash
# [TBD:e03]
# WikiText-103 validation set (100K tokens, 4 configs)
python3 v2/experiments/e03_capability_drift/run.py \
    --dataset wikitext --n_tokens 100000 --configs base,projector_off,projector_on,interrupt_on

# lm-eval-harness (HellaSwag, ARC-easy, MMLU-stem, 100 questions each)
python3 v2/experiments/e03_capability_drift/run.py \
    --dataset lm_eval --tasks hellaswag,arc_easy,mmlu_stem --n_per_task 100
```

### E04: ACT halt + K_max sweep
```bash
# [TBD:e04]
# K_max ∈ {2,4,8}, ponder loss λ ∈ {0.001,0.01,0.1}
python3 v2/experiments/e04_act_halt_kmax/run.py \
    --k_max 4 --lambda_ponder 0.01 --curriculum 1_to_kmax --seed 0 --n_test 300
```

### E05: Cross-model (Qwen3-1.7B / 4B / Llama-3.2-3B / Mistral-7B)
```bash
# [TBD:e05]
# Qwen3-1.7B (full anti-cheat, 1k steps)
python3 v2/experiments/e05_cross_model/run.py \
    --model Qwen/Qwen3-1.7B-Instruct --steps 1000 --n_train 1000 --n_test 300 --seed 0

# Qwen3-4B (extended 5k steps for H12 convergence test)
python3 v2/experiments/e05_cross_model/run.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --steps 5000 --n_train 5000 --n_test 300 --seed 0

# Llama-3.2-3B (cross-family, 1k steps)
python3 v2/experiments/e05_cross_model/run.py \
    --model meta-llama/Llama-3.2-3B-Instruct --steps 1000 --n_train 1000 --n_test 300 --seed 0

# Mistral-7B (if MPS 128GB permits, 2k steps)
python3 v2/experiments/e05_cross_model/run.py \
    --model mistralai/Mistral-7B-Instruct-v0.3 --steps 2000 --n_train 2000 --n_test 300 --seed 0
```

### E06: Relation-disjoint OOD
```bash
# [TBD:e06]
# Strict relation-level split (train relations ∩ test relations = ∅)
python3 v2/experiments/e06_relation_disjoint_ood/run.py \
    --split_mode relation_disjoint --n_train 1000 --n_test 200 --seed 0
```

### E07: Per-layer K-projector ablation
```bash
# [TBD:e07]
# Single-layer 9 / three-layer [9,15,21] independent P / three-layer shared P / three-layer per-layer rank-32 P
python3 v2/experiments/e07_per_layer_kproj/run.py \
    --config single_layer_9 --n_train 2000 --n_test 300 --seed 0

python3 v2/experiments/e07_per_layer_kproj/run.py \
    --config multi_layer_independent --layers 9,15,21 --n_train 2000 --n_test 300 --seed 0

python3 v2/experiments/e07_per_layer_kproj/run.py \
    --config multi_layer_shared --layers 9,15,21 --n_train 2000 --n_test 300 --seed 0

python3 v2/experiments/e07_per_layer_kproj/run.py \
    --config multi_layer_rank32 --layers 9,15,21 --rank 32 --n_train 2000 --n_test 300 --seed 0
```

### E08: Interrupt API demo
```bash
# [TBD:e08]
# Demo 1: GSM8K hard problems, oracle CoT hidden injection
python3 v2/experiments/e08_interrupt_api_demo/run.py \
    --demo gsm8k_oracle --n_test 50

# Demo 2: User text hint → encode → inject
python3 v2/experiments/e08_interrupt_api_demo/run.py \
    --demo text_hint --hint "use dynamic programming" --task algorithmic_qa --n_test 50
```

### E09: v1 AttnNativeBank resurrect (KProj integration)
```bash
# [TBD:e09]
# Integrate v2/core/kproj.py into v1 AttnNativeBank, run v1 test suite
python3 v2/experiments/e09_v1_resurrect_attn_native_bank/run.py \
    --test_suite v1_original --expect_bitequal_gate0

# Run Exp35b Φ1 oracle composition (k ∈ {1,10,100,1000})
python3 v2/experiments/e09_v1_resurrect_attn_native_bank/run.py \
    --test_suite exp35b_phi1_composition --k_sweep 1,10,100,1000
```

### E10: topK retrieval mechanism
```bash
# [TBD:e10]
# Compare all-attend (B2 default) vs topK ∈ {1,4,16,64}
python3 v2/experiments/e10_topk_retrieval/run.py \
    --topk all --n_test 300 --seed 0

for K in 1 4 16 64; do
    python3 v2/experiments/e10_topk_retrieval/run.py \
        --topk $K --retrieval_mode cosine --n_test 300 --seed 0
done
```

### E11: Dual-channel (auto-pause + interrupt)
```bash
# [TBD:e11]
# Auto-pause only / interrupt only / both
python3 v2/experiments/e11_dual_channel/run.py \
    --mode auto_pause_only --n_test 300 --seed 0

python3 v2/experiments/e11_dual_channel/run.py \
    --mode interrupt_only --n_test 300 --seed 0

python3 v2/experiments/e11_dual_channel/run.py \
    --mode dual_channel --n_test 300 --seed 0
```

### E12: Long-short coexistence
```bash
# [TBD:e12]
# LT only (8K preloaded) / ST only (pause-write) / LT+ST
python3 v2/experiments/e12_long_short_coexistence/run.py \
    --mode lt_only --n_preload 8192 --n_test 300 --seed 0

python3 v2/experiments/e12_long_short_coexistence/run.py \
    --mode st_only --pause_head_enabled --n_test 300 --seed 0

python3 v2/experiments/e12_long_short_coexistence/run.py \
    --mode lt_st --n_preload 8192 --pause_head_enabled --n_test 300 --seed 0
```

### E13: Multi-task capability
```bash
# [TBD:e13]
# Mixed batch: GSM8K + StrategyQA + NegationQA + CSQA + simple-QA
python3 v2/experiments/e13_multi_task_capability/run.py \
    --tasks gsm8k,strategyqa,negqa,csqa,simple_qa --n_per_task 200 --seed 0
```

### E14: Pause head training
```bash
# [TBD:e14]
# Unfreeze pause heads, train 1k steps with entropy bonus + λ_pause_reg
python3 v2/experiments/e14_pause_head_train/run.py \
    --steps 1000 --lambda_pause 0.01 --entropy_bonus 0.001 --n_train 2000 --n_test 300 --seed 0
```

### E15: Ponder curriculum
```bash
# [TBD:e15]
# K curriculum: epoch1 K=1→2, epoch2 K=2→4, epoch3 K=4→8
python3 v2/experiments/e15_ponder_curriculum/run.py \
    --curriculum 1_2_4_8 --epochs 3 --n_train 5000 --n_test 300 --seed 0
```

### E16: Bank capacity / forgetting
```bash
# [TBD:e16]
# Capacity C ∈ {64,256,1024,4096,16384}, eviction strategies: FIFO / LRU / random / score-based
for C in 64 256 1024 4096 16384; do
    for EVICT in fifo lru random score_based; do
        python3 v2/experiments/e16_bank_capacity_forgetting/run.py \
            --capacity $C --eviction $EVICT --n_test 300 --seed 0
    done
done
```

### E17: Negation robustness
```bash
# [TBD:e17]
# NegationQA + auto-constructed negation hard-negatives
python3 v2/experiments/e17_negation_robustness/run.py \
    --dataset negqa --augment hard_neg --n_test 300 --seed 0
```

### E18: Chained 2-hop reasoning
```bash
# [TBD:e18]
# 2-hop data from Exp35b subset, test K=2 + LT preload
python3 v2/experiments/e18_chained_2hop/run.py \
    --hops 2 --k_max 2 --n_preload 512 --n_test 300 --seed 0
```

### E19: Seed replication (B2 + e02 best + e04 best)
```bash
# [TBD:e19]
# B2 canonical, seed ∈ {0,1,2,3,4}
for S in 0 1 2 3 4; do
    python3 v2/experiments/e01_anticheat_b2/run.py --variant canonical --seed $S --n_test 120
done

# e02 best config (TBD from e02 results)
# for S in 0 1 2 3 4; do
#     python3 v2/experiments/e02_scale_matrix/run.py --config <BEST_FROM_E02> --seed $S --n_test 300
# done

# e04 best K_max + lambda_ponder (TBD from e04 results)
# for S in 0 1 2 3 4; do
#     python3 v2/experiments/e04_act_halt_kmax/run.py --config <BEST_FROM_E04> --seed $S --n_test 300
# done
```

**Note**: All JSON outputs are written to `v2/experiments/eXX_<name>/results/` with filenames `eXX_<variant>_seed<S>.json`. Bank source: `v1/experiments/atb_validation_v1/exp35b_memit_bank/data/bank.pt` (used by all experiments requiring preloaded LT memory).

---

## 8b. Next Steps (post-e11 wave-3)

Given the falsification in §1b, the experiment queue is reprioritized. The following three experiments are now **decisive** — together they determine whether v2 ships as "memory" (unlikely), "adapter" (likely), or "no defensible headline" (possible):

1. **e10 — top-K retrieval (HIGHEST PRIORITY).** Only mechanism that can recover content-sensitivity. Run real-bank vs random-bank vs collapsed-bank at top-K ∈ {1, 4, 16, 64} and at all-attend. **Decision rule**: if real-bank Δ at top-K=16 exceeds random-bank Δ by ≥ 2.0 NLL, retrieval is alive (memory framing partially recovers). Otherwise retrieval is dead.
2. **e13 — multi-task transfer.** Train on factual completion, evaluate on GSM8K / StrategyQA / NegQA / CSQA. **Decision rule**: if ≥ 2 tasks show Δ ≤ −0.5 vs base, the "adapter" reading is confirmed and v2 has a defensible (narrower) headline. If 0/4 tasks transfer, the projector is task-overfitted and even the adapter framing weakens.
3. **e06 — relation-disjoint OOD.** Strict relation-level split, Δ NLL on held-out relations. **Decision rule**: Δ ≤ −1.0 confirms broad adapter generalization; Δ → 0 means the effect is tied to relation structure shared between train and test (a different kind of leakage than H3 controls for).

All three should run at seed ∈ {0, 1, 2} from the outset — single-seed conclusions are no longer acceptable after e11 reversed the H7 reading at a different L2 norm.

**Deprioritized after e11**: e02 (scale matrix), e04 (ACT halt), e14 (pause heads), e15 (curriculum), e16 (capacity / forgetting). These all assume the memory-content claim; running them before e10 resolves retrieval-vs-adapter would burn compute on a framing that may already be dead.

**Still required regardless of e10 outcome**: e05 (cross-model) — adapter or memory, the mechanism must replicate beyond Qwen3-4B; e03 lm-eval portion — capability drift on benchmarks beyond WikiText-2; e19 (seed×3 replication) — all current single-seed numbers need variance bars before any verdict is sealed.

---

## 9. Termination Certificate

**This verdict will be sealed and considered final when ALL of the following conditions are met:**

1. **All experiment verdicts written**: `v2/verdicts/E01_VERDICT.md` through `E19_VERDICT.md` exist and follow the canonical template from V2_METHODOLOGY_DEBATE.md §5.1 (sections a-g: command, seeds, sample size, raw data path, numbers, verdict, caveat).

2. **All methodology documents complete**:
   - `v2/methodology/V2_METHODOLOGY_DEBATE.md` (self-deception checklist, anti-cheat philosophy, falsifier matrix) — ✅ EXISTS
   - `v2/methodology/V2_DIFFERENTIATION.md` (HNM vs RAG/LoRA/ICL/CoT/MEMIT/Constitutional AI) — ✅ EXISTS

3. **V1 tech debt closed**: `v2/tech_debt/V1_CLOSEOUT.md` complete, all 78 v1 todos dispositioned (wontfix-archive / roll-into-v2 / finalize-verdict / kill-blocked), SQL status updated to done/wontfix.

4. **H-matrix pass rate determined**: Final falsifier scoreboard (§3) shows ≥12/15 PASS (claim supported), or 9-11/15 PASS (grey zone, diagnostic follow-up required), or ≤8/15 PASS (claim refuted/revised).

5. **No abandonment triggers fired**: None of the 9 explicit abandonment conditions from V2_METHODOLOGY_DEBATE.md §4 have been triggered (e.g., capability collapse PPL >10%, no cross-model replication, seed variance ratio >0.30, etc.).

6. **User sign-off**: User explicitly approves this final verdict and authorizes sealing.

**Sealed by**: [NAME] on [DATE]

**Digital signature / commit hash**: [TBD — to be filled when sealed]

---

**END OF DRAFT**

*This document is a living skeleton. As experiments e01-e19 complete, sections 2-5 will be populated with actual numbers read from JSON outputs. The termination certificate (§9) will be filled when all conditions are met and user signs off. Until then, all `[TBD:eXX]` placeholders remain.*

---

**Document metadata**:
- **Created**: 2025-05-16 (v2 inception, post-Exp42 B2 pivot)
- **Last updated**: [TBD — update timestamp when first sealed]
- **Line count target**: 300-600 lines (current: ~580 lines)
- **Dependencies**: All e01-e19 experiment verdicts, V2_METHODOLOGY_DEBATE.md, V2_DIFFERENTIATION.md, V1_CLOSEOUT.md, master plan.md
- **Reproducibility**: Every experiment CLI is copy-pasteable and runs without editing. All raw data paths specified in §8.
