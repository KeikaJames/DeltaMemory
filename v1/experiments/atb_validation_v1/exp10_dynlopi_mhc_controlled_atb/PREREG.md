# Exp10 Pre-Registration: Dynamic LOPI + mHC Controlled ATB

**Experiment ID:** exp10_dynlopi_mhc_controlled_atb  
**Pre-registered:** 2026-05-09  
**Status:** LOCKED — do not edit after experiment start  

---

## 1. One-Sentence Hypothesis

> Exp10 tests whether the full ATB control stack — Dynamic LOPI v3.4 (layer/time routing),
> mHC-inspired bank shielding (spectral cap), and residual beta gating — can convert raw
> external-KV injection from a non-linear, alpha-sensitive perturbation into a controlled
> memory readout, as measured by CounterFact-style quantitative gap metrics and open-ended
> counterfactual generation quality.

---

## 2. Background and Motivation

- **Exp7** (n=807, Qwen3-4B, pre_rope): raw ATB correct_bank < random_kv. V-path dominance.
- **Exp8** (n=807, mHC κ=0.25): mHC-only cannot recover correct_bank selectivity.
  All κ values: gap ≤ 0. random_kv wins in all Phase B/C conditions.
- **Exp9** (beta gate + sep-softmax): tests residual smoothing in isolation.
- **Exp10**: closes the loop — if Dynamic LOPI routes injection to relevant layers/tokens,
  random KV content should be less exploitable even without correct K matching.

---

## 3. Research Questions

| ID | Question |
|----|----------|
| Q10.1 | Does Dynamic LOPI lower the margin advantage of random_kv / random_K_correct_V? |
| Q10.2 | Does mHC + Dynamic LOPI improve gap over mHC-only (Exp8)? |
| Q10.3 | Does beta=0.05 further reduce drift and value-path dominance? |
| Q10.4 | In counterfactual generation, does the control stack produce readable, low-leak answer shifts? |
| Q10.5 | Do LOPI γ_t and w(ℓ,t) actually gate (not always-open or always-closed)? |

---

## 4. Arms

| Arm | mHC | Dynamic LOPI | beta | Purpose |
|-----|-----|-------------|------|---------|
| A0_raw_atb | off | off | 1.0 | Raw baseline (Exp7/8 comparison) |
| A1_mhc_only | on (κ=0.25) | off | 1.0 | Exp8 exact control |
| A2_dynlopi_only | off | on | 1.0 | LOPI-only ablation |
| A3_mhc_dynlopi | on (κ=0.25) | on | 1.0 | mHC + LOPI |
| A4_mhc_dynlopi_beta | on (κ=0.25) | on | 0.05 | Full control stack |

Dynamic LOPI configuration (v3.4 retained form):
```
LOPIConfig(enabled=True, orthogonal=False, gaussian=True, derivative=True, profile_mode="auto")
```

---

## 5. Protocol

### 5.1 Dataset

- CounterFact 1k, pre-filtered (W.6 criterion), 807 eligible prompts
- `experiments/datasets/counterfact_1k.jsonl`

### 5.2 Bank Construction

- `bank_size=200` (1 target + 199 distractors, same protocol as Exp7/8/9)
- `bank_key_mode=pre_rope`, `value_scale_mode=auto_rms_cap`
- Write template: `"Fact: {subject} {phrase} {target_new}."`

### 5.3 Phase A — Smoke Grid

- `n_prompts=50`, `seeds=[0, 1, 2]`
- `alpha ∈ {0.05, 0.10, 0.20}` × all 5 arms
- Variants: `correct_bank`, `random_kv`, `random_K_correct_V`
- Selection metric: `gap_A = margin(correct_bank) - max(margin(random_kv), margin(random_K_correct_V))`
- Output: top-2 arms by best-alpha gap; best alpha for Phase B

### 5.4 Phase B — Confirm

- `n_prompts=200`, `seeds=[0, 1, 2]`
- Arms: **top-2 from Phase A + A0_raw_atb + A1_mhc_only** (deduplicated)
- Alpha: best from Phase A
- All 5 variants: `correct_bank`, `shuffled_bank`, `random_kv`, `correct_K_random_V`, `random_K_correct_V`

### 5.5 LOPI State Reset Rules

- **Benchmark scoring** (logp for target_new / target_true / rank / drift):
  LOPI state reset before **each independent forward**. Prevents prev_q from
  carrying over across prompt boundaries or between target evaluations.
- **Qualitative generation** (auto-regressive decode):
  LOPI state reset once per case × arm. State is NOT reset between decode steps.
  `prev_q_per_layer / prev_residual_norms / pending_residual_norms` remain live
  across tokens for correct derivative-gate behavior.

---

## 6. Pass / Fail Criteria

### PASS_STRONG

Phase B:
- Best controlled arm gap > A0_raw_atb gap AND > A1_mhc_only gap
- random_kv and random_K_correct_V margins reduced vs Exp8 baseline
- JS drift ≤ A0_raw_atb
- o_bank/o_seq ratio not inflated

Qualitative (Phase C):
- on-topic target_new_hit rate meaningfully above no_bank baseline
- off-topic_leak rate remains low (≤ 10%)
- prompt_echo rate ≤ 20%
- Answer coherence maintained

### PASS_STABILIZER

- gap ≤ 0 in Phase B
- But random controls are reduced vs A0 / A1
- Drift lower than raw ATB
- Qualitative: fewer leaks, more selective activation

Paper conclusion: *"Dynamic LOPI + mHC improved ATB injection stability, while row-level K
retrieval selectivity remains unresolved."*

### FAIL

- A3/A4 gap indistinguishable from A1_mhc_only
- random_kv continues to dominate in all conditions
- Qualitative: high prompt_echo / off-topic leak regardless of arm

Next step: bank_topk / K-projector / InfoNCE / target-slot diagnostics.

---

## 7. LOPI Causal Interpretation Rules

If A2_dynlopi_only (LOPI, no mHC) improves over A0_raw_atb:
→ LOPI routing alone has causal effect on selectivity.

If A3_mhc_dynlopi improves over both A1 and A2 individually:
→ mHC + LOPI have complementary effects.

If A4_mhc_dynlopi_beta further reduces drift over A3:
→ beta gating adds independent stability.

---

## 8. Registered Metrics

| Metric | Description |
|--------|-------------|
| `margin` | log p(target_new) - log p(target_true) |
| `gap` | margin(correct_bank) - max_control_margin |
| `JS_drift` | Symmetric JS divergence on 100 neutral windows |
| `KL_drift` | Forward KL on same neutral windows |
| `target_rank` | first-token rank of target_new |
| `recall_at_1` | target_rank == 0 |

Qualitative:
| Metric | Description |
|--------|-------------|
| `target_new_hit` | target_new string appears in generation |
| `target_true_hit` | target_true string still appears |
| `off_topic_leak` | counterfactual leaks into off-topic prompt |
| `prompt_echo` | write prompt text appears in generation |

---

*Pre-registration locked. See run.py for implementation.*
