# CHANGELOG

All notable changes to DeltaMemory are documented here, organised by
research stage. Older stages are summarised; the current stage is
documented in full enough to make the evidence and limits legible.

## Stage 15 — v3.1 cross-architecture attn-native bank

### v3.1 README and figure refresh
* `README.md` and `README.zh-CN.md` now use one unified DeltaMemory
  vocabulary and point only to the new v3.1 figure set.
* New dependency-free figure generator:
  `scripts/make_v31_readme_figures.py`.
* New SVG figures in `docs/figures/v31/`:
  architecture, counter-prior lift, Mac-vs-GB10 reproduction, DeepSeek-32B
  α sweep, and Gemma-4 dev_v31 recall context.

### Counter-prior intervention evidence
* `scripts/run_intervention_demo.py` supports `--false-facts` and records
  the target-token log-prob under B0 no-memory, B1 prompt-insertion, and
  v3 attn-bank conditions.
* Gemma-4-E2B with α=1.0: 5/5 counter-prior target lifts on GB10 CUDA and
  5/5 on Mac MPS.
* Qwen3-4B-Instruct with α=0.05: 5/5 counter-prior target lifts on GB10
  CUDA and 5/5 on Mac MPS.
* DeepSeek-R1-Distill-Qwen-32B α sweep at 0.05/0.10/0.20/0.30 is mixed:
  the identity-init bank is not enough to override all strong 32B priors.
  This is tracked as a real limitation and motivates a trained Qwen2-family
  K-projector.
* Raw transcripts are committed under `transcripts/v31_intervention/`.

### Cross-architecture α defaults
* `ArchAdapter.default_alpha` added:
  Gemma4=1.0, Qwen3=0.05, Llama/Qwen2=0.05, GLM-4=0.05.
* `scripts/run_intervention_demo.py --alpha` now defaults to the adapter's
  calibrated value.

## Stage 14 — v3, preregistered, frozen, with strict negative held-out test

### 14A — InfoNCE K-projector
* New `deltamemory/memory/k_projector.py`: per-attention-layer
  identity-initialised `Linear(d, d)` applied **only to bank keys**.
  `forward()` is functional (no module-graph mutation; safe under
  DDP / `torch.compile` / cross-arch).
* New `scripts/train_k_projector.py`: InfoNCE training on
  `eval/splits/train.jsonl` (104 facts × 5 paraphrases = 520 pairs).
  Final loss 2.89 → 0.95 across 12 epochs.
* Identity-init invariant: an attached-but-untrained projector is a
  bit-exact no-op; α=0 invariance preserved.

### 14B / 14C — capture-policy generalisation
* New `deltamemory/memory/capture_policy.py`: `CaptureSite` dataclass
  + `resolve_capture_sites()` for `period`, `address`, `multi`
  policies. Fallback to last token when an address span is unmatched.

### 14D — bank softmax temperature
* `attn_native_bank.py` now reads `bank.bank_temperature` (default 1.0,
  no-op) and divides bank scores by τ before the joint softmax.
  τ ≤ 0 raises `ValueError` (no silent clamp).

### 14E — ROME-style ridge-solve writer + bank-V rebuild
* New `deltamemory/memory/rome_writer.py`: ridge solve for the bank
  V slot; rebuild paths preserved by unit tests.

### 14F — frozen v3 config
* `deltamemory/configs/v3_frozen.yaml` with the K-projector's
  sha256 embedded. The frozen v3 = period-policy capture +
  trained K-projector + τ=1.0 + write_alpha=1.0 + read_alpha=1.0.

### Preregistration + holdout split
* New `eval/holdout_split.py` (deterministic stratified split) with
  `--check` mode against the committed manifest.
* `eval/splits/{train,dev,test}.jsonl` + `manifest.json` (sha256-pinned).
* `docs/preregistration.md` binds H1–H7, gates, statistical procedure
  (paired Wilcoxon, Holm-Bonferroni at α=0.05, 2k bootstrap CIs).

### Phase G — held-out test eval (one-shot)
* `scripts/run_stage14_test_eval.py`: 5 conditions
  (B0 / B1 prompt / B2 RAG-oracle / v2 / v3 frozen).
* `reports/cleanroom/stage14_test_gemma4_e2b/{REPORT.md, summary.json, stats.json}`.
* **Headline (Gemma-4-E2B / MPS bf16)**:
  * v3 = 0.2778, B0 = 0.3590 (Δ=−0.081, p=0.0074, **H1 REJECTED**).
  * v3 − v2 = +0.278 (p<0.001, H1b confirmed).
  * v3 − B1 prompt = −0.380 (H2 REJECTED).
* The test split is now CONSUMED. Any future v3.x changes go through
  an Amendment block in `docs/preregistration.md`.

### Phase H — reproducibility infrastructure
* `repro_v3.sh`: one-liner end-to-end reproduction of every artifact.
* `scripts/make_figures.py` → 6 figures in
  `reports/cleanroom/figures/`.
* `scripts/demo_chat.py`: interactive REPL demo (baseline / prompt /
  v3 side-by-side at the next-token logit).
* `examples/demo_prompts.md`: factual recall + long-form fidelity
  example prompts.

### PR #2 review fixes
* `KProjectorBank.forward`: stop mutating `self.layers` on the hot path.
* `bank_temperature ≤ 0` raises `ValueError`.
* `write_fact()` uses the proper `CaptureSite` dataclass instead of
  `type("S", (), …)()`.
* `_eval_condition` drops unused `write_alpha` parameter.
* `train_k_projector.py`: drop unused `math` / `DataLoader` imports.
* `docs/design.md` ArchAdapter table is explicit about Gemma-4-only
  support today; Qwen3 / Llama / GLM-4 marked as v3.1 work.

### Methodology amendment (written-down)
* Per-fact dev/test sign flip (+8.1 pp → −8.1 pp) is documented in
  the Phase G report as the textbook overfit-selection signature.
* v3.1 prerequisites (must be cleared before any new freeze):
  ≥ 2,500 training pairs across ≥ 30 relations; cross-relation hard
  negatives mandatory; structural fix for softmax dilution at N≥30
  (top-k / cosine / separate bank-only head); two-stage held-out gate
  (dev + a second validation split) before any "frozen" claim.
* Comparison frame: until (1)–(4) are satisfied, the only supported
  claim DeltaMemory makes is "matches or beats prompt-insertion at
  equal compute on a preregistered held-out split". B1 = 0.658.

## Stage 13 — AttentionNative DeltaMemory v2

* Zero learnable parameters: bank K/V are exactly what the model emits
  during a one-shot write pass.
* α=0 bit-equality + locality bit-equality preserved.
* KV-shared layer routing fix (Stage 13D): single-fact target rank
  41 → 9 on the unit gate.
* Strict negative on chat recall (Stage 13F): write-time K and read-time
  Q live in different K-space regions; zero-shot softmax can't bridge
  the gap. This motivated Stage 14.

## Stage 12 — adversarial cross-model validation
* Single-model harness completed; multi-model deferred until Stage 13's
  K-space gap is addressed and an `ArchAdapter` exists per family.

## Stage 11 — retraining + conv baselines (NVIDIA GB10 Blackwell)
* Address-encoder upgrades (`prompt_hidden`, multi-layer concat).
* Recall@1 = 1.000 on real LAMA-TREx across 7 relations, 3 seeds, swap
  paired-flip 0.989 ± 0.010, head-to-head wins over RAG / IKE / SFT-LoRA.

## Stage 10 —顶会级 adversarial validation
* 3-seed paired-bootstrap CIs.

## Stages 8–9 — closed-book address-keyed fast-weight bank
* Closed-book recall, swap-binding, no-leakage gates pass at N up to
  4096 on a frozen Gemma-4-E2B.

## Stages 0–7 — in-context binding + LM-head LoRA
* End-to-end Q/V residual + LM-head rank-4 LoRA hits oracle upper
  bound on LAMA factual cards.

---

Maintained by KeikaJames. Co-authored by Copilot CLI in early stages.
