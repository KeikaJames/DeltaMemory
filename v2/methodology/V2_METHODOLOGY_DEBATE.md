# V2 Methodology Debate — Rigor, Skepticism, and Reproducibility

> **Purpose**: Enumerate every cheap explanation we can devise for Exp42 Phase B2's Δ=−5.83 NLL drop, document how each is falsified, and establish the statistical + methodological floor to avoid self-deception.

---

## 1. Why we are skeptical of B2 (Δ=−5.83)

Exp42 Phase B2 showed: Qwen3-4B frozen, 512 Exp35b b-vectors in layer-9 bank, rank-64 K-projector trained 200 steps → test NLL **12.13 → 6.30 (Δ=−5.83)**; random-bank control Δ=−0.02.

**Every cheap explanation we can think of:**

| H | Cheap Explanation | How Falsified | Pass Criteria |
|---|---|---|---|
| **H1** | Projector learns vocab bias (any b → high-freq token distribution) | e01-1: eval with bank-off (frozen trained projector, empty bank) | NLL ≥ base − 0.05 |
| **H2** | b-vectors already encode target embeddings; projector is decoder shortcut | e01-2: row-level shuffle b dimensions (preserves L2 norm, destroys semantic structure) | NLL degrades ≥ +4.0 vs canonical |
| **H3** | Train/test entity or relation leakage | e01-3: strict entity-disjoint + relation-disjoint splits (train ∩ test = ∅ on both axes) | Δ NLL still ≤ −1.0 |
| **H4** | Random-bank control is buggy (randomization bypassed, same data used) | e01-4: zero bank.slots directly (explicit null) | NLL ≈ base ±0.02 |
| **H5** | One bank entry is enough (pseudo-retrieval, projector ignores bank size) | e01-5: N_preload sweep {1, 2, 4, 16, 64, 256, 512, 2048, 8192} × seed×3 | NLL monotonically improves with N |
| **H6** | Layer-9 selection is accidental (any layer would work) | e01-6: layer sweep {3, 9, 15, 21, 27, 33} × seed×3 | ≥3 layers achieve Δ ≤ −1.0 |
| **H7** | Random bank + projector training can achieve same performance | e01-7: train random-bank from scratch 200 steps | random stays ≥ 4.0 NLL above real bank |
| **H8** | General LM capability is destroyed (overfitting to fact-recall) | e01-8 + e03: WikiText-103 PPL + lm-eval (HellaSwag, ARC, MMLU) | rel PPL drift ≤ 5%, acc drop ≤ 2pp |
| **H9** | Single-model fluke (Qwen3-4B architecture quirk) | e01-9 + e05: cross-model (Qwen3-1.7B, Llama-3.2-3B, Mistral-7B) | ≥1 model achieves Δ ≤ −1.0 |
| **H10** | Bank_gate learns degenerate on/off (always 0 or always 1) | e01-10: gate histogram + per-layer entropy (H = −Σ p log p) | gate selectivity entropy > 0.3 nats |
| **H11** | K-projector rank-64 is over-parameterized (memorizing test set) | e02: projector rank sweep {8, 16, 32, 64, 128} + N_train {120, 1k, 5k} | lower rank (16/32) still Δ ≤ −2.0 |
| **H12** | 200 steps is cherry-picked convergence point | e02: step sweep {50, 100, 200, 500, 1000, 5000} | improvement plateaus after 200, no U-shape |
| **H13** | Test set is accidentally similar to train (train/test correlation) | e06: relation-disjoint OOD (train relations ∩ test relations = ∅) | Δ NLL ≤ −0.5 on OOD |
| **H14** | Effect is tied to specific prompt format or tokenization | e13: multi-task eval (GSM8K, StrategyQA, NegQA, CSQA) | ≥2 tasks show Δ ≤ −0.5 |
| **H15** | Learning rate 2e-4 is tuned to this exact data | e02: lr sweep {5e-5, 1e-4, 2e-4, 5e-4, 1e-3} | ≥2 lr values achieve Δ ≤ −3.0 |

**Total falsifiers**: 15. **Pass threshold**: ≥12/15 → main claim stands; ≤8/15 → main claim rejected; 9-11/15 → grey zone, demand diagnostic follow-up.

---

## 2. Self-Deception Checklist

**12 concrete anti-patterns an over-eager AI experimenter would commit:**

| # | Bad Pattern | Rule We Follow |
|---|---|---|
| 1 | Eyeball numbers from terminal, round in head, report "~6.3" | Always read final numbers from JSON; print 4 decimals in verdict |
| 2 | Run control once, assume it generalizes | Every control must run seed×3 minimum; report mean ± std |
| 3 | "Random control looks similar" → skip it | Random/zero/shuffle controls are **mandatory** for every core claim |
| 4 | Use same random seed for train and control | Control must use **different random instantiation** (not same seed) |
| 5 | Test set N=10, declare victory | N_test ≥ 120 (3× Phase B2's N=40); if N<120, verdict includes CAVEAT |
| 6 | "Loss went down" → assume test performance improved | Training loss is **diagnostic only**; verdict decision uses held-out test NLL |
| 7 | Train on full dataset, "hold out" 10% post-hoc | Split BEFORE any data loading; use dataset's native train/test if available |
| 8 | Layer selection: "tried 2 layers, picked the best" | If layer is tuned, report tuning procedure + ablation on ≥3 non-tuned layers |
| 9 | Shuffle indices but not data (off-by-one bugs) | Shuffle the actual tensor rows OR the index list, verify with md5(data[:5]) |
| 10 | "Epoch 5 looks good" → stop early, don't check epoch 10 | Run to predefined step budget; if early-stop, report full loss curve + why |
| 11 | Compare train NLL vs base's test NLL | **All comparisons use same split** (test-vs-test, train-vs-train) |
| 12 | "I'll check WikiText later" → never check | Capability drift (WikiText PPL + lm-eval) is **blocking** for any trainable-param experiment |

**Commitment**: If we catch ourselves violating any of these, we **immediately discard that result** and re-run cleanly.

---

## 3. Statistical Floor

### 3.1 Minimum Sample Sizes

| Metric | Minimum N | Minimum Seeds | Rationale |
|---|---|---|---|
| Test eval (NLL on answer) | **120** | **3** | B2 used N=40; we 3× it. Seed×3 catches optimizer/data-order flukes. |
| Capability drift (WikiText PPL) | **100K tokens** | **1** (deterministic) | Enough to estimate PPL within ±0.05 relative error. |
| Ablation (layer sweep, etc.) | **60 per config** | **3** | Half of main test N, still enough for t-test at p<0.05 with typical Δ. |
| Control (random bank, zero bank) | **same as test N** | **3** | Control must be **equally powered** as main experiment. |

### 3.2 Go/No-Go from std/mean Ratio

After seed×3, compute Δ_mean and Δ_std across seeds:

- **std(Δ) / |mean(Δ)| < 0.15** → robust, proceed
- **0.15 ≤ ratio < 0.30** → borderline, report with warning, add seed×5
- **ratio ≥ 0.30** → too noisy, increase N_test or abandon config

Example (hypothetical):
- Seed 0: Δ = −5.83
- Seed 1: Δ = −5.91
- Seed 2: Δ = −5.76
- Mean Δ = −5.83, std = 0.075 → ratio = 0.075 / 5.83 = **0.013** ✅ robust

If we saw Δ = {−5.8, −2.1, −8.9} → ratio = 0.58 → **reject config**, demand re-tuning.

### 3.3 Significance Tests (When Needed)

For pairwise comparisons (e.g., config A vs config B on same test set):
- Use **paired t-test** (per-item Δ, not per-seed aggregate) if items independent
- Report **p-value** and **effect size (Cohen's d)** in verdict
- Threshold: **p < 0.01** (Bonferroni-correct if >5 comparisons)

**Do NOT** rely solely on p-value: a Δ = −0.03 with p=0.001 is statistically significant but scientifically meaningless.

---

## 4. When to Abandon

**Explicit abandonment triggers** (any one sufficient to declare HNM line dead):

### 4.1 Core Falsification Failures

1. **H-matrix collapse**: <8/15 cheap explanations falsified after full e01+e02 suite
   → **Verdict**: Phase B2 is artifact, not genuine memory integration; revise central claim to "K-projector enables limited static retrieval but not hippocampus-style dynamics"

2. **No cross-model replication**: e05 shows ≥3 models all Δ > −0.5
   → **Verdict**: Qwen3-4B fluke; HNM does not generalize across architectures

3. **Capability collapse**: WikiText PPL increase >10% relative OR lm-eval acc drop >5pp
   → **Verdict**: Memory mechanism breaks general LM; unacceptable trade-off

4. **No pause-write benefit**: e14 (pause head training) shows auto-pause Δ ≤ +0.1 vs frozen-pause
   → **Verdict**: Working memory (ST bank via pause) is null; downgrade claim to "static LT memory only"

5. **No multi-round benefit**: e04 (K_max sweep) shows K=2,4,8 all equivalent to K=1
   → **Verdict**: Multi-round mechanism is vestigial; single-round attention suffices

### 4.2 Engineering Intractability

6. **Gradient pathology**: Any experiment requiring >10K steps to converge OR lr sweep shows no stable regime
   → **Verdict**: Projector training is brittle; not viable for production

7. **Memory explosion**: e16 (capacity) shows C=1024 necessary for Δ ≤ −1.0 but C=1024 incurs >3× latency
   → **Verdict**: Compute cost prohibitive for claimed benefit

### 4.3 Reproducibility Collapse

8. **Seed variance explosion**: e19 shows std(Δ)/|mean(Δ)| > 0.30 after seed×5
   → **Verdict**: Effect is not robust; publish negative result with full data

9. **User-reported non-replication**: External researcher (or user on different hardware) cannot reproduce B2 within Δ ± 1.0 NLL
   → **Verdict**: Pause all claims, enter debugging mode, publish replication kit

### 4.4 What Does NOT Trigger Abandonment

- A single experiment (e.g., e08 interrupt demo) failing → other experiments can compensate
- One model in e05 failing → as long as ≥2 succeed, claim survives
- Minor capability drift (2-3% PPL increase) → acceptable if memory benefit is large (Δ < −3.0)

**Procedure on trigger**: Write `V2_NEGATIVE_RESULT.md`, archive all data, commit with conclusion, notify user, **stop all further HNM experiments**.

---

## 5. Reproducibility Contract

**Every experiment verdict MUST include, in this exact order:**

### 5.1 Canonical Template

```markdown
## Experiment eXX: <Title>

### (a) Command
```bash
python3 v2/experiments/eXX_*/run.py \
    --device mps \
    --seed 0 \
    --n_test 120 \
    --<param> <value>
```

### (b) Seeds
- Seed 0, 1, 2 (or 0-4 for critical experiments)

### (c) Sample Size
- N_train = <num>
- N_test = <num>
- N_preload = <num> (if applicable)

### (d) Raw Data Path
- JSON: `v2/experiments/eXX_*/results/eXX_<config>_seed<S>.json`
- Bank: `v1/experiments/atb_validation_v1/exp35b_memit_bank/data/bank.pt` (if used)

### (e) Numbers (read from JSON, 4 decimals)
| Metric | Seed 0 | Seed 1 | Seed 2 | Mean ± Std |
|---|---|---|---|---|
| base NLL | 12.1310 | 12.1290 | 12.1330 | 12.131 ± 0.002 |
| config A NLL | 6.3050 | 6.2880 | 6.3120 | 6.302 ± 0.012 |
| Δ A | −5.826 | −5.841 | −5.821 | **−5.829 ± 0.010** |
| config B NLL | 12.1120 | 12.1050 | 12.1180 | 12.112 ± 0.007 |
| Δ B | −0.019 | −0.024 | −0.015 | **−0.019 ± 0.005** |

### (f) Verdict
**PASS** / **FAIL** / **MARGINAL**

Reason: <one-sentence summary of why this experiment supports or refutes the hypothesis>

### (g) Caveat
<any limitations, e.g., "N_test=60 (below floor of 120)", "single model only", "layer pre-selected by B2", "no cross-architecture test">

If no caveat, write: **None.**
```

### 5.2 Enforcement

- If a verdict is missing any of (a)-(g), it is **invalid** and does not count toward the H-matrix pass rate.
- User will reject any "trust me, I ran it" claims without the JSON path and raw numbers.
- Command in (a) must be **copy-pasteable** and run to completion without editing (no placeholder `<path>`).

---

## 6. Commitment to Future Self / External Reviewers

This document is a **binding contract**. If, at any point during v2 experiments, we:
- Skip a control because "it seems obvious it will fail"
- Use N=30 because "120 takes too long"
- Cherry-pick the best seed and hide the other two
- Tune hyperparams on test set and report as if it were held-out
- Fail to check WikiText PPL after training
- Report "NLL ≈ 6.3" instead of reading the JSON

...then we have **violated this contract** and must discard that experiment's results.

**The user has been burned by AI agent over-eagerness multiple times.** We compensate by:
1. Overcounting cheap explanations (15 instead of 10)
2. Triple the sample size (N=120 vs B2's 40)
3. Mandatory seed×3 (vs B2's seed×1)
4. Mandatory controls for every claim (random, zero, shuffle)
5. Mandatory capability drift check (WikiText + lm-eval)
6. Explicit abandonment rules (so we know when to stop digging)

**If this level of rigor prevents us from making grand claims, so be it.** A narrow, well-supported claim beats a broad, shaky one.

---

*Last updated: v2 inception (post-Exp42 B2 pivot)*
