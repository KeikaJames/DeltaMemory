# Honest Scoreboard: What DeltaMemory Can and Cannot Prove

**Phase U.5** — DeltaMemory theory documentation.

> **Tone policy.** This document does not sugarcoat. Every claim below is grounded in a specific experiment or code inspection. Unverified claims are explicitly labelled as such.

---

## 1. mHC Shield — Current Verdict

### 1.1 Q2 Sweep: 168 cells

The Q2 sweep ran mHC shield (on/off) across a grid of models, $\alpha$ values, and bank sizes, measuring mean NLL drift on a neutral text set. Summary:

| Model | Has `v_norm` | mHC result | Notes |
|---|---|---|---|
| Gemma-4-E2B |  | **Full PASS** | Drift controlled at all tested $\alpha$. Only model where mHC alone was sufficient. |
| Qwen2.5-0.5B |  | H1 fail | Drift floor 2–12 nats. mHC reduced drift partially; did not clear threshold. |
| GLM-4-9B-Chat |  | H1 fail | Same pattern as Qwen. |
| DeepSeek (dense) |  | H1 fail | Same pattern. |

**Root cause analysis.** The H1 failures on no-`v_norm` models are not evidence that mHC is mathematically wrong. The column cap correctly bounds $\sigma_{\max}(W_{:,T:}) \leq \kappa$. However, on Qwen/GLM, $\|M_V\|$ is 5–10× larger than on Gemma (no native `v_norm`), so the injection energy $\kappa \cdot \alpha \|M_V\|$ remains large even with the column cap active. mHC without V-scale is insufficient for no-`v_norm` families. **R-7 (V-scale) was introduced precisely because of this failure pattern.**

### 1.2 R-7 Partial Fix

After R-7, V-scale + mHC together were tested on Qwen2.5-0.5B and observed to reduce drift to an acceptable level for the tested $\alpha$ values. This is **one data point on one model**. The combined fix has not been validated on Qwen3-4B, GLM-4-9B, Llama-3.2-1B, or any other no-`v_norm` family.

**Current confidence level:** mHC + V-scale is a sound theoretical combination and has one empirical supporting result. It is **not validated across the architecture family**.

---

## 2. mHC × MoE: Never Tested

**Status: zero experiments.** mHC has never been run on Mixtral, Qwen3-MoE, DeepSeek-V3, or any other MoE architecture. The global column-cap formula is theoretically wrong for MoE (see `docs/theory/mhc_moe.md`), but the magnitude of the error is unknown.

This is the sharpest unanswered question in the mHC component. Phase W.5 is the only path to an answer.

---

## 3. LOPI — Current Verdict

### 3.1 Sole Strong Evidence: GPT-2-medium at $\alpha = 8$

The R-3 ablation (630 cells) tested five LOPI variants (A0–A4) on GPT-2-medium across $\alpha \in \{0.25, 0.5, 1, 2, 4, 8\}$. The result:

- **At $\alpha = 8$ (catastrophic injection regime):** variant A4 (Gaussian + derivative, no orthogonal projection) reduced mean NLL drift by **65–95%** relative to A0 (no LOPI).
- **At $\alpha \in [1, 5]$ (production regime):** no variant consistently outperformed A0. Some variants made things slightly worse.

**This is the only strong evidence that any LOPI component works.** It is strong in that the effect size (65–95%) is large enough to be unambiguous. It is weak in that: (a) $\alpha = 8$ is not a production value, (b) the evidence comes from a single model family (GPT-2), and (c) the mechanism (why does A4 help at $\alpha = 8$ but not at $\alpha = 2$?) is not understood.

### 3.2 Production $\alpha$ (1–5): Auto vs. Static Regression

Phase S introduced `profile_mode="auto"` to replace Gemma-only constants with architecture-profiled Z-score signals. The S-7 experiment on Qwen2.5-0.5B compared auto vs. static mode.

**Result: auto did not outperform static.** Mean drift under auto mode was comparable to or worse than static mode for production $\alpha$ values.

**Potential root causes (unconfirmed):**
- $N = 10$ profiling corpus is too small; $\mu_{\text{arch}} = \arg\max_\ell \sigma_{\text{base}}(\ell)$ may be unstable.
- Z-score depth signal may be noisy with small $N$, causing $\mu_t$ to drift erratically.
- Qwen2.5-0.5B may have flat $\sigma_{\text{base}}$ profile (low CV), making $\eta_\sigma = 1.0$ (no shrinkage), and the argmax may be degenerate.

None of these are confirmed. Phase W.2-Q3 is designed to test the $\mu_{\text{arch}}$ stability directly.

### 3.3 R-5.1 Chat Implant: The Hardest Failure

Phase R-5.1 tested knowledge implant: write 5 facts to the bank, then probe whether the model generates the correct answer (greedy rank-1) for each fact.

**Result on Gemma-4-E2B: 0/5 strict implant.** At $\alpha \in \{8, 10\}$, 1/5 partial pass was observed (log-prob of correct answer increased but did not reach greedy rank-1). At production $\alpha$ values, zero facts were correctly retrieved by greedy decoding.

This is the hardest evidence against LOPI's production readiness. An injection method that can increase the log-probability of a fact without making it the model's top prediction is not useful for knowledge implant tasks. The gap between "log-prob improves" and "greedy output changes" may reflect:

1. The correct answer's prior log-prob is very low; a small $\alpha \cdot \|M_V\|$ perturbation is insufficient to overcome the prior.
2. The attention routing ($W_{:,T:}$) is diluted across many tokens; no single query attends strongly enough to the bank.
3. LOPI's $M_\perp$ projection removes exactly the component of $M_V$ that would shift the model's output distribution toward the correct answer (the alignment prior hypothesis from U.3).

None of these are confirmed diagnoses. They are failure-mode hypotheses for Phase W.3 to test.

---

## 4. Summary Scoreboard

| Component | Claim | Evidence level | Notes |
|---|---|---|---|
| mHC math (column cap bounds $\sigma_{\max}$) | Correct |  Proven (linear algebra, unit tests) | No empirical doubt. |
| mHC on Gemma (with `v_norm`) | Works |  Q2 168-cell full PASS | Only one model family. |
| mHC on no-`v_norm` models | Partial |  H1 fail without V-scale | R-7 fixes Qwen2.5-0.5B; others untested. |
| mHC + V-scale cross-arch | Unvalidated |  1 data point (Qwen2.5-0.5B) | Must run W.1 full grid. |
| mHC on MoE | Unknown |  Never run | Formula is wrong; no data. |
| LOPI at $\alpha = 8$ (GPT-2) | Works |  65–95% drift reduction (R-3) | Narrow regime, single model. |
| LOPI at production $\alpha$ (1–5) | Weak |  No consistent win in R-3 or S-7 | Active area of investigation. |
| LOPI auto-mode vs. static | Regression |  S-7 Qwen2.5-0.5B: auto ≤ static | Root cause unknown. |
| Chat implant (5 facts, Gemma) | Failed |  R-5.1: 0/5 strict implant | Log-prob improves; greedy does not. |
| Cross-arch $\alpha$-linearity | Unvalidated |  No experiment | Requires both V-scale + direction alignment (W.4). |

---

## 5. What Phase W Must Prove

Phase W is the proving ground. The minimum bar for claiming any component is "production-ready" is:

1. **mHC + V-scale:** 5-model drift grid (W.1) with shield+V-scale holding mean drift $\leq 1.0$ nats at $\alpha = 2$ across all 5 dense architectures. Current status: 1/5 validated.

2. **LOPI production regime:** At least one LOPI variant outperforming A0 on $\geq 3/5$ models at $\geq 3/5$ production $\alpha$ values in W.2 ablation. Current status: 0 production-regime wins.

3. **Chat implant:** At least 3/5 facts reaching greedy rank-1 on at least 2 model families in W.6 (CounterFact benchmark). Current status: 0/5 on Gemma.

**As of Phase U, none of these bars have been cleared.** The mathematical foundations (Sections 1–3 of this document set) are sound. The empirical validation is incomplete.

---

## 6. Closing Statement

DeltaMemory's approach — inference-time K/V bank injection into frozen LLMs, with spectral regularisation (mHC) and adaptive modulation (LOPI) — is theoretically coherent and has isolated empirical support in narrow regimes. It is not production-ready. Phase W exists to either build the evidence base that justifies production deployment, or to identify what needs to be replaced.

The correct scientific posture is: **the method is promising, the math is right, and the evidence is not there yet.**

---

*References:* DeltaMemory Q2 sweep results; Phase R-3 ablation `reports/cleanroom/lopi_v33/FINDINGS.md`; Phase R-5.1 chat implant results; Phase S-7 Qwen2.5-0.5B auto/static comparison.
