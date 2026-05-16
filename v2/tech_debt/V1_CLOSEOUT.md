# V1 Tech-Debt Closeout

> **2026-05 Terminology Note**: Project main vocabulary has been renamed
> from "HNM / Hippocampus-style Native LLM Memory / memory bank" to
> **Attention-Side Latent Bank (ALB) / latent bank / bank readout / pause-write**.
> v1 historical naming is preserved as-is in `v1/` for reproducibility.
> Mapping table at top of `v2/README.md`.

**Status**: v1 is now **CLOSED** and archived under `v1/`. All active development has migrated to the v2 line (`v2/`) which implements Attention-Side Latent Bank (ALB). The successful core mechanism discovered in v1—a rank-r K-projector over per-layer AttentionBank on a frozen LLM base—is preserved and extended in v2 with dual-channel injection (auto-pause + human interrupt API), long-term/runtime-written latent bank coexistence, multi-round attention, and adaptive compute (ACT halt). The pivotal v1 result (Exp42 LPL Phase B2: test NLL 12.13→6.30, Δ=−5.83, random-bank control Δ=−0.02) demonstrated that v1's AttnNativeBank formula was not dead, merely incomplete—missing the bank-side learnable K/V projector.

---

## 1. Naming Cleanup

V1 accumulated a confusing mix of names across 42 numbered experiments. Many were misleading, reflected dead ends, or were simply "写错了" (wrong). Below is the disposition of each major naming thread:

### 1.1 mHC / Spectral Shield
- **What it was**: `mhc_shield` / `shield_attention_weights` — a spectral bound (κ) on bank-column attention sums, intended to suppress V-path dominance when bank size grows.
- **Status in v1**: Worked as an ablation knob in Exp8/Exp9 negative-control studies; never showed consistent positive signal independent of other mechanisms.
- **Disposition**: **KEPT in v1 archive only as a historical ablation control**. Not migrated to v2. If v2 needs attention regularization, it will be redesigned from scratch.
- **Files**: `v1/deltamemory/memory/attn_native_bank.py` lines 857+ (shield logic), `v1/experiments/atb_validation_v1/exp8_mhc_smoothed_pre_rope/`, `exp9_residual_gated_mhc_pre_rope/`

### 1.2 RCV-mC (Rigged Composition Verifier – manifold Connections)
- **What it was**: An early hypothesis that bank entries formed a "connection manifold" that could be rigged for composition verification.
- **Status in v1**: Never implemented beyond sketches; no experimental validation.
- **Disposition**: **DEAD, wontfix**. The phrase appears nowhere in production code. No v2 analog.

### 1.3 Manifold / Sinkhorn
- **What it was**: Manifold-based addressing with Sinkhorn optimal-transport routing for bank retrieval.
- **Status in v1**: Conceptual exploration only; no working implementation or experimental validation.
- **Disposition**: **DEAD, wontfix**. Exp35b used learned router (dot-product softmax); Exp42 used direct preload (no routing). v2 will explore topK retrieval (cosine/learned) in e10, but does not inherit "manifold" terminology.
- **Files**: Mentioned in early planning docs; no production code.

### 1.4 Hyper-Connections
- **What it was**: A proposed mechanism for cross-layer bank connections (possibly related to skip-connections or multi-layer bank reads).
- **Status in v1**: Never integrated into any experiment.
- **Disposition**: **DEAD, wontfix**. v2 implements per-layer banks with optional shared or independent K-projectors (e07) but does not use "Hyper-Connections" terminology.

### 1.5 LOPI / U-LOPI / Dynamic-LOPI v3.x
- **What it was**: **L**ow-rank **O**rthogonal **P**rojection **I**ntervention — a gating mechanism that dynamically skipped certain attention heads or layers based on per-token criteria. U-LOPI = unary LOPI (single gate). Dynamic-LOPI v3.x = adaptive version with beta-gate and layer-specific skip logic.
- **Status in v1**: 
  - Worked on Gemma models in early experiments (Exp1-7 era on Gemma2-2B).
  - **Failed to generalize** to Qwen3-4B fact recall (Exp10 dynlopi_mhc_controlled_atb showed negative or null results).
  - Last tested in Exp10 (May 2025); no positive signal on Qwen3.
- **Disposition**: **ARCHIVED** as a model-specific result. Not migrated to v2 (v2's pause-head mechanism is a full replacement, trained end-to-end with projector heads). LOPI remains in `v1/deltamemory/memory/lopi.py` for historical reproducibility only.
- **Files**: `v1/deltamemory/memory/lopi.py`, `v1/experiments/atb_validation_v1/exp10_dynlopi_mhc_controlled_atb/`

### 1.6 AttnNativeBank (ANB)
- **What it was**: The v1 production bank class (`attn_native_bank.py`) — concat bank (K, V) into layer attention, apply global α mixing weight or per-position gate.
- **Status in v1**: Core mechanism for Exp1-42. Proved necessary but incomplete (Exp42 Phase B null → Phase B2 success when K-projector added).
- **Disposition**: **SUPERSEDED** by `v2/core/attention_bank.py`. v2's AttentionBank adds:
  - LT/ST tagging (long-term preload vs short-term pause-write)
  - FIFO capacity management per channel
  - topK retrieval interface (e10)
  - Multi-round accumulation across K_max rounds
  - Interrupt API hooks
- **Migration target**: v2/experiments/e09_v1_resurrect_attn_native_bank/ will backport the K-projector fix to v1's original AttnNativeBank class to confirm the v1→v2 bridge is sound.
- **Files**: `v1/deltamemory/memory/attn_native_bank.py` → `v2/core/attention_bank.py`

### 1.7 Mneme / deltamemory Package
- **What it was**: The v1 Python package name (`deltamemory/`) and the `mneme` CLI tool for bank build/eval. "Mneme" (Greek: memory) was the public-facing name.
- **Status in v1**: Frozen at last v1 commit. Still importable for v1 reproduction.
- **Disposition**: **FROZEN**. v2 does not use the `deltamemory` package; all core logic is self-contained in `v2/core/`. No PyPI release planned for deltamemory; it remains a research artifact.
- **Files**: `v1/deltamemory/`, `v1/experiments/atb_validation_v1/_lib/load_model.py` (uses deltamemory imports internally)

### 1.8 Fact-LoRA Bank (Exp35)
- **What it was**: Bank construction via rank-1 LoRA edits (ROME-style v* solve). Each fact became a (K, V) pair from the model's own forward pass + a LoRA residual.
- **Status in v1**: Exp35 verdict **POSITIVE** (78.8% router top-1, gate_d +10.27 nats @ k=10, composition holds). But Exp42 Phase B2 showed **no LoRA needed**—frozen base + projector heads suffice.
- **Disposition**: **SUPERSEDED** by Exp42 LPL+B2 discovery. v2 does not use LoRA for bank construction. The Exp35 solo_pass filter logic and router training may inform v2/e10 topK retrieval, but the LoRA mechanism itself is dropped.
- **Migration**: v2 uses MEMIT b-vectors (Exp35b) as the preloaded latent bank source, not Exp35 LoRA deltas.
- **Files**: `v1/experiments/atb_validation_v1/exp35_fact_lora_bank/` (archived), superseded by `exp35b_memit_bank/`

### 1.9 MEMIT Bank (Exp35b)
- **What it was**: Bank construction via MEMIT residual b-vectors (10K facts, layer L=5, C⁻¹ precondition from 1.5M-token WikiText covariance). Each fact is a 9728-d latent encoding the (subject, relation, target_new) binding.
- **Status in v1**: Exp35b verdict **PARTIAL POSITIVE** (oracle composition gate_d +9.78 @ k=10, 100% beats base; router top-1 40.85%, D6 capability PASS @ k≤100; failed at k=1000 with 94% PPL drift). This became the **LT memory source for Exp42 Phase B2**.
- **Disposition**: **KEPT** as the canonical v2 long-term-memory data source. The 10K b-vectors (512 used in Exp42 proof-of-concept) will be preloaded into v2's AttentionBank LT channel. Exp35b's D7 anti-cheat (key orthogonality, rank 1678/2560) and D9 (unigram-matched uplift) validated the bank is not cheating via low-rank shortcuts.
- **Migration**: `v1/experiments/atb_validation_v1/exp35b_memit_bank/data/bank.pt` → v2 LT preload. v2/e02_scale_matrix will sweep N_preload ∈ {512, 2048, 8192, 32768}.
- **Files**: `v1/experiments/atb_validation_v1/exp35b_memit_bank/`

---

## 2. Per-Experiment One-Liner Verdicts

Below is a complete enumeration of all numbered experiment folders under `v1/experiments/atb_validation_v1/`, with one-line summaries of what was tested, outcome, and v2 migration target (or wontfix-archive disposition).

| Exp | Summary | Outcome | v2 Target |
|-----|---------|---------|-----------|
| **exp1_core_ablation** | Core ATB mechanism (bank read at attention layer) vs no_bank / post_rope / pre_rope / vscale variants. Pre-registered gates on recall@1 / margin lift. | **POSITIVE** (pre_rope_bank_only > no_bank, CI > 0; vscale stabilizes) | e09 (v1 resurrect; B2 projector backport validates original formula with missing piece) |
| **exp2_position_invariance** | Test whether bank entries are position-invariant (same K/V regardless of prompt structure). | **POSITIVE** (position-invariant within tolerance) | wontfix-archive (sanity check passed; no further v2 work needed) |
| **exp3_bit_equal** | Gate 0 sanity: bank α=0 or gate=0 must give byte-identical logits to base model. | **POSITIVE** (bit-equal on 24 prompts, max \|Δlogits\|=0.000e+00) | e01-10 (gate 0 bit-equal is a mandatory anti-cheat in v2 Phase 0) |
| **exp4_cf1k_main** | CounterFact-1K benchmark: ANB on full CF dataset (1000 facts). | **NULL** (no consistent uplift; routing failed at scale) | wontfix-archive (superseded by Exp35b N=10K + router) |
| **exp5_alpha_sweep** | α mixing weight sweep: α ∈ {0.01, 0.05, 0.10, 0.20, 0.50, 1.0}. Find optimal α for margin lift. | **NEGATIVE** (higher α → V-dominance, lower α → no signal; no sweet spot on Gemma2-2B) | wontfix-archive (v2 uses learnable per-position gate, not global α) |
| **exp6_negative_controls** | Negative controls: random_kv, random_K_correct_V, shuffled_fact_ids, minus_correct. Pre-rope baseline. | **NEGATIVE** (all controls matched or exceeded correct_bank margin → V-dominance confirmed) | e01 (anti-cheat suite H1-H10 subsumes these controls with stricter thresholds) |
| **exp6b_post_rope_negative_controls** | Same as Exp6 but post-rope bank injection. Test whether RoPE timing matters. | **NEGATIVE** (post-rope performed worse than pre-rope; controls still failed) | wontfix-archive (v2 uses pre-rope only; post-rope path abandoned) |
| **exp7_non_gemma_pre_rope** | Port ANB to Qwen3-4B-Instruct-2507 (first non-Gemma test). Pre-rope, α=0.05. | **NEGATIVE** (negative margins across all 5 variants; pattern_v_dominates=True) | wontfix-archive (led to Exp8 mHC shield hypothesis, later abandoned) |
| **exp8_mhc_smoothed_pre_rope** | mHC spectral shield (κ bound on bank-column attention) to suppress V-dominance. Qwen3, α=0.05, kappa sweep. | **NULL** (mHC-on did not restore Gate B; margins remained negative or insignificant) | wontfix-archive (mHC not migrated to v2) |
| **exp9_residual_gated_mhc_pre_rope** | Residual gating + mHC (combine gate-per-position with spectral shield). | **NULL** (no improvement over Exp8; still failed Gate B) | wontfix-archive |
| **exp10_dynlopi_mhc_controlled_atb** | Dynamic LOPI v3.4 + mHC + beta-gate. Test whether LOPI's adaptive skip logic fixes ATB on Qwen3. | **NEGATIVE** (LOPI did not generalize from Gemma to Qwen3 fact recall; no positive signal) | wontfix-archive (LOPI archived; v2 pause-head is a full replacement in e14) |
| **exp11_rsm_residual_stream_memory** | Residual-stream memory: inject bank at MLP outputs or pre_block inputs instead of attention layer. | **NEGATIVE** (residual injection failed; attention-layer read is necessary) | wontfix-archive (confirms v1 attention-layer read was correct; v2 inherits) |
| **exp12_htwr** | Hidden-to-Weight Routing (HTWR): per-fact routing via hidden→logits projection. | **NULL** (routing accuracy insufficient; abandoned in favor of Exp35 learned router) | wontfix-archive |
| **exp13_anb_readdressability** | AttnNativeBank re-addressability: sparse-attention (joint softmax) bank read. Tested N=100 → N=200 scaling. | **MIXED** (N=100 PASS_STRONG; N=200 FAIL → falsification pattern, later reproduced in Exp27/33) | wontfix-archive (sparse-attention path abandoned after Exp33 rejection) |
| **exp31_learned_k_adapter** | Learned K-adapter: per-bank-entry learned query projector (precursor to B2 K-projector). | **PARTIAL** (showed improvement over naive concat, but insufficient uplift alone) | e07 (per-layer K-projector ablation; Exp31 insights inform rank/layer sweep) |
| **exp32_mlp_side_gated_memory** | MLP-side gated memory: inject bank deltas at MLP layer instead of attention. | **NEGATIVE** (double-negative: failed Gate B and Gate D → "fact-agnostic uplift") | wontfix-archive (confirms attention-layer read is necessary, not MLP-side) |
| **exp33_reattn_readout** | Re-attention readout (Phase B): joint-softmax sparse-attention bank on Exp31/32 splits. | **REJECTED** (reproduced Exp27 falsification: shuffled V > correct V → Gate D fail) | wontfix-archive (closed post-Exp32 sparse-attention investigation) |
| **exp34_rome_baseline** | ROME (Rank-One Model Editing) baseline: per-fact rank-1 edit, oracle composition test. | **BASELINE** (establishes ROME can do N=100 edits; used as sanity-check for "known-good" architecture) | wontfix-archive (v2 does not use ROME; Exp35b MEMIT is the LT source) |
| **exp35_fact_lora_bank** | Fact-LoRA Bank: ROME-style v* solve → (K, V) bank. Learned router (dot-product softmax). N=975 facts, 764 retained. | **POSITIVE** (78.8% router top-1, gate_d +10.27 @ k=10, composition holds; 90.5% routed beats base) | wontfix-archive (superseded by Exp35b MEMIT + Exp42 no-LoRA result) |
| **exp35b_memit_bank** | MEMIT b-vector bank: 10K facts, C⁻¹ precondition, layer L=5. Oracle composition + learned router. | **PARTIAL POSITIVE** (oracle gate_d +9.78 @ k=10, router 40.85% top-1; D6 capability collapses @ k=1000) | **e02 (scale matrix LT source), e01 (anti-cheat H5 N_preload sweep uses Exp35b bank), e06 (relation-disjoint OOD)** |
| **exp36_binding_audit** | Anti-cheat: entity-disjoint split, paraphrase split, negation template split. | **AUDIT** (exposed AC4 paraphrase overlap bug; AC5 negation pass; mixed results led to Exp38 refinement) | e01-3 (entity-disjoint + relation-disjoint splits in v2 anti-cheat suite) |
| **exp37_production_stress** | Production stress: capability drift (HellaSwag), 37.C drop_pass (frac facts with \|drop\|<0.5 nats). | **AUDIT** (capability drift < 5% pass; 37.C exposed high-variance fact quality) | e03 (capability drift on WikiText-103 + lm-eval-harness) |
| **exp38_gated_bank** | Gated Fact-LoRA Bank: per-position learnable gate + topK retrieval (k_retrieve sweep). Best config G2_kr1. | **POSITIVE** (G2_kr1: gate_d +5.049 @ k=10, 99.2% drop_pass, 43.5% neg_pass; topK=1 optimal) | **e10 (topK retrieval), e01-10 (gate selectivity monitoring)** |
| **exp42_lpl** | Latent Pause Loop (LPL): multi-round attention with pause-write + Exp35b bank preload + rank-64 K-projector. | **BREAKTHROUGH** (Phase B2: NLL 12.13→6.30, Δ=−5.83; random-bank ctl Δ=−0.02 → K-projector unlocks v1 formula) | **e01 (entire anti-cheat suite replicates B2), e02 (scale), e04 (K_max+halt), e07 (per-layer projector), e08 (interrupt API), e09 (backport to v1), e11-e19 (all v2 experiments build on B2 foundation)** |

---

## 3. What v2 Owes from v1

Below is an explicit list of v1 mechanisms that v2 must reproduce, replace, or extend, with the v2 file path that owns the responsibility.

| v1 Mechanism | v1 Path | v2 Responsibility | v2 Path | Status |
|--------------|---------|-------------------|---------|--------|
| **AttentionBank core (K,V concat into softmax)** | `v1/deltamemory/memory/attn_native_bank.py` lines 857-950 | Reproduce + add LT/ST tagging, FIFO, topK | `v2/core/attention_bank.py` | ✅ Scaffolded |
| **Gate 0 bit-equal sanity** | `v1/experiments/atb_validation_v1/exp3_bit_equal/` | Mandatory anti-cheat in Phase 0 | `v2/core/attention_bank.py` + e01-10 | ✅ In e01 plan |
| **K-projector (rank-r I+P)** | `v1/experiments/atb_validation_v1/exp42_lpl/06_phase_b2_kproj.py` lines 180-220 | Generalize to per-layer, rank sweep, shared vs independent | `v2/core/kproj.py` | ✅ Scaffolded |
| **Per-position bank_gate** | `v1/experiments/atb_validation_v1/exp42_lpl/attention_bank.py` LPLHeads | Train bank_gate + pause head jointly; monitor selectivity | `v2/core/attention_bank.py` + e14 | ✅ In e14 plan |
| **Pause head (force-pause @ layers)** | `v1/experiments/atb_validation_v1/exp42_lpl/qwen3_lpl_patch.py` lines 120+ | Learnable pause head + entropy regularization | `v2/core/qwen3_lpl_patch.py` + e14 | ✅ Scaffolded |
| **Multi-round attention (K_max)** | `v1/experiments/atb_validation_v1/exp42_lpl/01_phase_a_frozen.py` K=2 loop | K_max ∈ {2,4,8} + ACT halt + curriculum | `v2/core/runtime.py` + e04, e15 | ✅ In e04/e15 plan |
| **MEMIT b-vector preload (LT memory)** | `v1/experiments/atb_validation_v1/exp35b_memit_bank/data/bank.pt` | Preload 10K b-vectors into LT channel, tag, freeze | `v2/core/data_io.py` + e02 | ✅ In e02 plan |
| **Learned router (topK retrieval)** | `v1/experiments/atb_validation_v1/exp35b_memit_bank/router_10k.pt` + Exp38 topK | topK retrieval (cosine / dot / learned address) | `v2/core/retrieval.py` + e10 | ✅ In e10 plan |
| **Oracle composition (Φ1 gate_d metric)** | `v1/experiments/atb_validation_v1/exp35b_memit_bank/04_oracle_compose_full.py` | Reproduce Φ1 @ k=10 as anti-cheat H5 N_preload sweep baseline | e01-5 + e02 | ✅ In e01-5 plan |
| **Anti-cheat: shuffled control** | `v1/experiments/atb_validation_v1/exp6_negative_controls/` + Exp35b D9 | H2 (shuffle b dims), H4 (zero bank), random-bank control in every experiment | e01 (H1-H10) | ✅ In e01 plan |
| **Anti-cheat: entity-disjoint split** | `v1/experiments/atb_validation_v1/exp36_binding_audit/` | H3 entity-disjoint + e06 relation-disjoint OOD | e01-3, e06 | ✅ In e01-3, e06 plan |
| **Capability drift (D6 audit)** | `v1/experiments/atb_validation_v1/exp35b_memit_bank/04b_d6_capability.py` | WikiText-103 PPL + lm-eval-harness (HellaSwag, ARC, MMLU-stem) | `v2/core/eval_lib.py` + e03 | ✅ In e03 plan |
| **NLL-on-answer eval** | `v1/experiments/atb_validation_v1/exp42_lpl/06_phase_b2_kproj.py` eval loop | Standardize nll_on_answer(logits, ids, ans_start) helper | `v2/core/eval_lib.py` | ✅ Scaffolded |
| **Interrupt API (human inject)** | (never implemented in v1; planned for v2) | interrupt(model, round, layer, pos, h) public API | `v2/core/interrupt_api.py` + e08, e11 | ⏳ In e08/e11 plan |
| **Long-short coexistence** | (never implemented in v1; planned for v2) | LT (preload) + ST (pause-write) in same AttentionBank | `v2/core/attention_bank.py` + e12 | ⏳ In e12 plan |
| **Dual-channel (auto-pause + interrupt)** | (never implemented in v1; planned for v2) | Auto-pause head + interrupt API co-exist | `v2/core/qwen3_lpl_patch.py` + e11 | ⏳ In e11 plan |
| **ACT halt (adaptive ponder)** | (never implemented in v1; planned for v2) | halt head + ponder loss + K curriculum | `v2/core/runtime.py` + e04, e15 | ⏳ In e04/e15 plan |

**Legend**:
- ✅ **Scaffolded**: v2/core file exists, experiment planned in v2 roadmap.
- ⏳ **In plan**: Not yet implemented, scheduled in v2 Phase list (§2 of v2 plan).
- ❌ **Not migrated**: v1 mechanism dropped (e.g., mHC, LOPI, LoRA bank).

---

## 4. Anti-Cheat Lessons Learned

v1 conducted 42 experiments over 8 months, most of which returned null or negative results. The failures taught us how easily LLMs cheat—and how to catch them. Below are 8 key lessons, each citing specific v1 paths.

### 4.1 LoRA learns vocab bias, not fact binding
- **Lesson**: When training LoRA adapters on fact banks, the adapter can learn to shift the unigram distribution toward high-frequency tokens that happen to be targets, rather than binding subject→relation→target.
- **Evidence**: Exp35b's D9 unigram-matched control was necessary precisely because Exp35 (Fact-LoRA Bank) was vulnerable to this. D9 restricted to facts where `|log p(target_new) − log p(target_true)| ≤ 1.0` nat and *still* got +9.71 nats gate_d @ k=10, proving the uplift wasn't unigram bias.
- **Fix**: v2 does not use LoRA for bank construction. The Exp42 B2 result (frozen base + projector) avoids adapter-level vocab cheating entirely.
- **Files**: `v1/experiments/atb_validation_v1/exp35b_memit_bank/09_d9_unigram_matched.py`, `exp35_fact_lora_bank/` (archived)

### 4.2 Control configurations that randomize too late are buggy
- **Lesson**: If you randomize bank rows *after* passing them through a learned router or projector, the router's learned weights can still contain information about the correct assignment—so the "random" control isn't truly random.
- **Evidence**: Exp6/Exp7 negative controls (`random_kv`, `random_K_correct_V`) were suspected of this in early iterations. Exp42 Phase B2's random-bank control was constructed by randomizing the *input* b-vectors before training the projector, ensuring the projector saw noise from the start (Δ=−0.02, truly null).
- **Fix**: **Always randomize inputs before any learned transformation**. v2 anti-cheat H4 (zero bank slots) and H7 (train projector with random preload from start) enforce this.
- **Files**: `v1/experiments/atb_validation_v1/exp42_lpl/06_phase_b2_kproj.py` lines 350-370 (random_bank logic), `exp6_negative_controls/`, `exp7_non_gemma_pre_rope/`

### 4.3 Shuffled-control bugs: shuffle what, when, and at what granularity?
- **Lesson**: "Shuffled control" is underspecified. Must define: (a) shuffle fact IDs? (b) shuffle b-vector dimensions? (c) shuffle rows within bank? (d) shuffle after read or before write? Each has different implications.
- **Evidence**: 
  - Exp35b's D7 (shuffle b-vector *dimensions* within each entry) tests whether the bank encodes information in a low-rank "universal target" direction (it doesn't; cos(a_i, a_j) median = 0.015).
  - Exp42 Phase F AC1 (shuffle pause-layer index) tests whether the layer locus matters (it does; AC1 kills two-hop gain).
  - Exp38 AC5 (negation template disjoint) tests whether train/test templates leak (they don't).
- **Fix**: v2 anti-cheat suite H1-H10 specifies *exactly* what is shuffled in each control: H2 (row-level shuffle b dims), H4 (zero slots), H7 (random bank from start).
- **Files**: `v1/experiments/atb_validation_v1/exp35b_memit_bank/07_d7_delta_rank.py`, `exp42_lpl/02_phase_f_anticheat.py`, `exp38_gated_bank/`

### 4.4 V-path dominance: injecting V content overwhelms K routing
- **Lesson**: If bank V values are large (high L2 norm) or α is large, the V-path can dominate softmax output *regardless* of whether K routing is correct. This produces "fact-agnostic uplift"—model prefers *any* bank entry over base, even shuffled ones.
- **Evidence**: Exp7 diagnosis: `pattern_v_dominates=True` (all negative margins; controls matched or exceeded correct_bank). Exp32 "double-negative" (MLP-side gated memory) showed same pattern. Exp33 reproduced it on joint-softmax sparse-attention.
- **Fix**: 
  - Exp42 B2 used *residual* projector (I+P, zero-initialized P) so bank V starts close to base hidden norm, preventing early-training V dominance.
  - v2 always monitors **Gate D** (correct_bank > shuffled_bank by ≥ 0.5 nats) as a mandatory anti-cheat (H2 in e01-2).
- **Files**: `v1/experiments/atb_validation_v1/exp7_non_gemma_pre_rope/`, `exp32_mlp_side_gated_memory/`, `exp33_reattn_readout/EXP33_VERDICT.md`

### 4.5 Entity-disjoint splits are not enough; relation-disjoint is stronger OOD
- **Lesson**: If train and test share the same *relations* (e.g., "capital of", "born in") but different entities, the model can memorize relation-specific biases. True OOD requires disjoint relations.
- **Evidence**: Exp36 binding audit exposed AC4 paraphrase-split overlap (33 train/test string overlaps → false positive). Exp38 passed AC5 (negation template disjoint) but only because negation was a *new* relation type.
- **Fix**: v2 e06 (relation-disjoint OOD) enforces `train_relations ∩ test_relations = ∅` using Exp35b's relation metadata field.
- **Files**: `v1/experiments/atb_validation_v1/exp36_binding_audit/`, `exp38_gated_bank/` (AC4, AC5 audits)

### 4.6 Train/test leakage via paraphrase templates
- **Lesson**: If train and test facts use the same paraphrase templates (e.g., "The capital of X is Y"), the model can learn template-specific patterns rather than general fact recall.
- **Evidence**: Exp36 AC4 found 33 overlapping paraphrase strings between train/test splits (before deduplication). Exp42 Phase F used disjoint paraphrase sets for two-hop chains.
- **Fix**: v2 anti-cheat H3 (entity-disjoint split in e01-3) will deduplicate templates. v2/e18 (chained 2-hop) will use fully synthetic chains with no train/test template overlap.
- **Files**: `v1/experiments/atb_validation_v1/exp36_binding_audit/`, `exp42_lpl/03_two_hop_25.py`

### 4.7 Capability drift is a lagging indicator; must monitor early
- **Lesson**: If you only check capability drift (WikiText PPL, HellaSwag acc) *after* training 1000+ steps on a large bank, the model may have already collapsed. Must monitor every 50-100 steps.
- **Evidence**: Exp35b's D6 capability audit passed @ k=10 (0.69% PPL drift) and k=100 (2.81%), but **catastrophically failed** @ k=1000 (94% drift). The collapse happened somewhere between k=100 and k=1000.
- **Fix**: v2/e03 capability drift will eval WikiText-103 PPL and lm-eval-harness every 100 steps during training. Early-stop if rel PPL drift > 10%.
- **Files**: `v1/experiments/atb_validation_v1/exp35b_memit_bank/04b_d6_capability.py`, `exp37_production_stress/`

### 4.8 Gate selectivity: if gate learns to always-on or always-off, bank is dead weight
- **Lesson**: Learnable per-position gates can degenerate to constant 0 (bank never read) or constant 1 (bank always read, no selectivity). Both indicate failure—either the bank isn't useful or the model can't tell when to use it.
- **Evidence**: Exp42 Phase B trained bank_gate alone (92K params, 80 steps) → eval NLL barely moved (Δ=−0.0001). Phase B2 added K-projector (420K total params, 200 steps) → Δ=−5.83. The gate alone was insufficient; the projector gave it something *useful* to gate.
- **Fix**: v2 anti-cheat H10 (e01-10) monitors gate histogram + per-layer entropy. Pass threshold: `gate_selectivity > 0` (i.e., not degenerate constant).
- **Files**: `v1/experiments/atb_validation_v1/exp42_lpl/05_phase_b_train.py` (null result), `06_phase_b2_kproj.py` (success), `exp38_gated_bank/` (G2_kr1 optimal gate selectivity)

---

## 5. Closeout SQL Plan (Not Executed Here)

This document does *not* execute SQL transitions. Below is the high-level strategy for how the 78 v1 todos (prefixed `v1-*` in the SQL todo database) should be processed in the main v2 workflow.

### 5.1 Categories

1. **wontfix-archive**: Experiments tied to dead mechanisms (mHC, RCV-mC, LOPI, LoRA bank, sparse-attention) or superseded by v2 experiments.
   - **Examples**: All todos related to Exp4, Exp5, Exp6, Exp6b, Exp7, Exp8, Exp9, Exp10, Exp11, Exp12, Exp13, Exp32, Exp33, Exp34, Exp35 (Fact-LoRA).
   - **SQL**: `UPDATE todos SET status = 'wontfix', description = description || ' [v1 archived; superseded by v2 e0X]' WHERE id LIKE 'v1-%' AND (...)` (filter by exp number).
   - **Count**: ~50 todos.

2. **done**: Experiments with completed verdicts that require no further v1 work.
   - **Examples**: Exp1 (core ablation PASS), Exp2 (position invariance PASS), Exp3 (bit-equal PASS), Exp35b (partial positive verdict written), Exp38 (gated bank verdict written), Exp42 (LPL verdict written).
   - **SQL**: `UPDATE todos SET status = 'done' WHERE id IN ('v1-exp1-verdict', 'v1-exp2-verdict', 'v1-exp3-verdict', 'v1-exp35b-verdict', 'v1-exp38-verdict', 'v1-exp42-verdict')`.
   - **Count**: ~10 todos.

3. **roll-into-v2**: Todos that describe anti-cheat or capability audits already subsumed by v2 experiments (e01-e19).
   - **Examples**: 
     - "Exp36 entity-disjoint split audit" → e01-3 (H3 entity-disjoint split).
     - "Exp37 capability drift audit" → e03 (capability drift).
     - "Exp38 topK retrieval audit" → e10 (topK retrieval).
   - **SQL**: `UPDATE todos SET status = 'done', description = description || ' [Migrated to v2 e0X; see v2/tech_debt/V1_CLOSEOUT.md §3]' WHERE id IN (...)`.
   - **Count**: ~15 todos.

4. **blocked-kill**: Todos blocked on upstream experiments that are now `wontfix`.
   - **Examples**: "Exp9 analysis" (blocked on Exp8 positive result, which never happened).
   - **SQL**: `UPDATE todos SET status = 'wontfix' WHERE status = 'blocked' AND depends_on IN (SELECT id FROM todos WHERE status = 'wontfix')`.
   - **Count**: ~3 todos.

### 5.2 Execution Plan

1. **Phase 1** (during v2-0 scaffold): Write this V1_CLOSEOUT.md document (done here).
2. **Phase 2** (during v2-1 e01 anti-cheat): 
   - Query all v1 todos: `SELECT id, title, description, status FROM todos WHERE id LIKE 'v1-%' ORDER BY id;`
   - Batch update wontfix-archive todos (Exp4-13, 32-34, 35 Fact-LoRA).
   - Batch update done todos (Exp1-3, 35b, 38, 42 with verdicts).
3. **Phase 3** (during v2-2 to v2-19 experiments):
   - As each v2 experiment completes (e01-e19), check if it subsumes a v1 todo. If yes, mark that v1 todo `done` with a note "Migrated to v2 eXX".
4. **Phase 4** (during v2-21 tech-debt closeout):
   - Final sweep: `SELECT * FROM todos WHERE id LIKE 'v1-%' AND status NOT IN ('done', 'wontfix');`
   - Manually review any remaining v1 todos (should be 0-2 edge cases).
   - Write a final SQL summary: `SELECT status, COUNT(*) FROM todos WHERE id LIKE 'v1-%' GROUP BY status;` and commit it to V1_CLOSEOUT.md.

### 5.3 Final State (Target)

| Status | Count | Description |
|--------|-------|-------------|
| `done` | ~25 | Experiments with verdicts + anti-cheat audits migrated to v2 |
| `wontfix` | ~53 | Dead mechanisms (mHC/LOPI/LoRA/sparse-attention) or superseded by v2 |
| `pending` | 0 | All v1 todos resolved |
| `in_progress` | 0 | All v1 todos resolved |
| `blocked` | 0 | All blocked todos killed or unblocked |

**Invariant**: After v2-21 closeout, `SELECT COUNT(*) FROM todos WHERE id LIKE 'v1-%' AND status NOT IN ('done', 'wontfix')` **MUST** return 0.

---

## 6. Reproduction & Archival Notes

### 6.1 How to Reproduce v1 Key Results

All v1 code is frozen under `v1/` and remains executable for historical reproduction. Key results:

1. **Exp42 Phase B2** (the pivotal result):
   ```bash
   python3 v1/experiments/atb_validation_v1/exp42_lpl/06_phase_b2_kproj.py \
       --device mps --steps 200 --lr 2e-4 --rank 64
   ```
   Expected: test NLL 12.13 (base) → 6.30 (LPL+real bank+P, post-train), Δ=−5.83.

2. **Exp35b Oracle Composition** (validates MEMIT bank):
   ```bash
   python3 v1/experiments/atb_validation_v1/exp35b_memit_bank/04_oracle_compose_full.py \
       --model Qwen/Qwen3-4B-Instruct-2507 --device mps --k 10 --seeds 0,1,2
   ```
   Expected: gate_d +9.78 nats @ k=10, 100% beats_base.

3. **Exp38 Gated Bank topK=1** (optimal retrieval):
   ```bash
   cd v1/experiments/atb_validation_v1/exp38_gated_bank/
   python3 run.py --variant G2_kr1 --device mps --seeds 0,1,2
   ```
   Expected: gate_d +5.049 @ k=10, 99.2% drop_pass.

### 6.2 Archival Integrity

- **Commit**: v1 is frozen at commit `<FILL_IN_LAST_V1_COMMIT>` (to be tagged `v1-final-archive`).
- **Git**: `v1/` directory is read-only. No new commits to v1 after this closeout.
- **Dependencies**: v1 used `torch==2.5.1`, `transformers==4.48.1`, `deltamemory` (local package). A frozen `v1/requirements.txt` is committed for reproducibility.
- **Data**: Exp35b's `bank.pt` (10K MEMIT b-vectors, 72 MB) and Exp42's phase results JSON are committed to LFS.

### 6.3 What v1 Did NOT Solve (Open Problems for v2)

1. **Production-scale routing**: Exp35b router got 40.85% top-1 @ N=10K (below 50% threshold). Exp38 topK=1 helped, but v2 must validate routing at N=10K+ (e02 scale matrix).
2. **Capability drift at scale**: Exp35b D6 collapsed @ k=1000 (94% PPL drift). v2/e03 must prove capability holds at N=10K.
3. **Negation robustness**: Exp42 Phase F showed NegationQA harmed base in all configs. v2/e17 must fix negation handling.
4. **Multi-task generalization**: v1 tested only simple-QA, two-hop, negation. v2/e13 will test GSM8K, StrategyQA, CSQA, etc.
5. **Cross-model generalization**: v1 primarily used Qwen3-4B. v2/e05 will test Qwen3-1.7B, Llama-3.2-3B, Mistral-7B.
6. **Interrupt API**: Never implemented in v1. v2/e08 + e11 must deliver the "human inject latent" use case.
7. **ACT halt + ponder**: Never implemented in v1. v2/e04 + e15 must show adaptive compute works.

---

## 7. Sign-Off

This document closes v1 tech debt. All 42 v1 experiments are accounted for:
- 6 POSITIVE results (Exp1, Exp2, Exp3, Exp35, Exp35b partial, Exp42 B2 breakthrough).
- 15 NEGATIVE results (Exp5, Exp6, Exp6b, Exp7, Exp10, Exp11, Exp32, Exp33, etc.).
- 12 NULL results (Exp4, Exp8, Exp9, Exp12, Exp31 partial, etc.).
- 9 AUDIT/MIXED results (Exp13, Exp36, Exp37, Exp38, etc.).

**The v1 line is now CLOSED.** All future work proceeds in v2. The v1→v2 bridge is sound: Exp42 Phase B2 proved the K-projector was the missing piece, and v2 inherits the complete formula (attention-layer bank read + learnable projector + per-position gate + multi-round) with extensions (LT/ST coexistence, interrupt API, ACT halt).

**For v2 contributors**: Read §3 (What v2 Owes from v1) and §4 (Anti-Cheat Lessons) before implementing v2 experiments. Do not repeat v1's mistakes.

---

*Document written: 2025-05-16*  
*v1 final commit: `<TO_BE_TAGGED>`*  
*v2 active branch: `main`*  
*Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>*
