# LOPI: Layer-Orthogonal-Projection Injection — Full Derivation and Critique

**Phase U.3** — DeltaMemory theory documentation.

---

## 1. Overview

LOPI (Layer-Orthogonal-Projection Injection) replaces the naive bank readout

$$\text{out}_{\text{bank}} = W_{:,T:} \cdot (\alpha M_V)$$

with a three-component modulated injection:

$$\text{out}_{\text{bank}}^{\text{LOPI}} = \gamma_t \cdot w(\ell, t) \cdot M_\perp$$

where $M_\perp$ is the component of the bank value orthogonal to the native context readout, $w(\ell, t)$ is a Gaussian over layer index, and $\gamma_t$ is a derivative-triggered gate. The design intent is: inject only *novel* information ($M_\perp$), inject it at the *most relevant* layer ($w$), and inject it only when the topic is *changing* ($\gamma_t$).

The implementation lives in `deltamemory/memory/lopi.py` and `deltamemory/memory/lopi_profiler.py`.

---

## 2. Formula Table with Derivation and Known Issues

The following table reproduces the complete LOPI formula chain. Each row gives: the original formula as specified in the preregistration, its current implementation status, and known issues that Phase W.2 must resolve.

| Symbol | Formula | Implementation | Known Issues |
|---|---|---|---|
| $\Delta_Q$ | $\|Q_t - Q_{t-1}\|_2$ | `lopi.py` L242–243 | KV-cache prefill: $Q_{t-1}$ does not exist at $t=0$ or after session reset. Handled by returning $\gamma_t = 1$ (session-boundary fallback, L239–241). |
| $\gamma_t$ | $\sigma\!\left(k(\Delta_Q - \theta)\right),\; k=5,\; \theta=0.5$ | `lopi.py` L245–246 | $\theta = 0.5$ silently assumes unit-norm Q vectors. Qwen3 Q norms are ~5× larger than Gemma; the gate will be near-constant 1 for those models, making the derivative gate a no-op. These constants are Gemma-only defaults that must be profiled in W.2. |
| $\hat{d}_t$ | Static: $\sigma\!\left(\kappa_d \cdot (\bar{N}_{t-1}/\texttt{norm\_base} - 1)\right)$; Auto: Z-score of avg layer norm | `lopi.py` L323–331 (static), L318–320 (auto) | `norm_base = 10.0` is hard-coded for Gemma-4-E2B (L113). In auto mode (`profile_mode="auto"`) the depth signal is replaced by a Z-score relative to the per-layer $(\mu_{\text{base}}, \sigma_{\text{base}})$ from `LOPIProfile`, making it architecture-agnostic. |
| $\mu_{\text{arch}}$ | $\arg\max_\ell \sigma_{\text{base}}(\ell)$ | `lopi_profiler.py` L108–110 | `_argmax_low_tiebreak` breaks ties toward lower index (L105–110). Risk: $\arg\max$ is sensitive to small fluctuations in a $N=10$ corpus. The selected layer may differ by $\pm L/4$ if the corpus is unrepresentative. Phase W.2-Q3 tests this directly. |
| $\mu_t$ | Original: $\mu_t = L(0.3 + 0.5 \hat{d}_t)$; Auto: $\mu_t = \mu_{\text{arch}} + c(\hat{d}_t - 0.5)L$ | `lopi.py` L331 (static), L320 (auto) | **Deliberate departure from preregistration.** In auto mode, $\mu_{\text{arch}}$ anchors the Gaussian at the empirically most-variable layer, not a fixed fraction of depth. Static mode preserves the original formula bit-exactly for regression. This departure must be validated in W.2. |
| $\sigma_t$ | Original: $\sigma_t = (L/6) e^{-\beta \bar{\sigma}_{\max}}$; Auto: $\sigma_t = (L/6) \eta_\sigma$ | `lopi.py` L332 (static), L321 (auto) | **$\bar{\sigma}_{\max}$ is dead code in production (B3).** `LOPIState.mhc_sigma_max_running` (L189) is never populated by the bank patcher. The static-mode $\sigma_t$ formula therefore reduces to $(L/6)e^0 = L/6$ always. Auto mode replaces this with $\eta_\sigma \in \{0.7, 1.0\}$ from `LOPIProfile.eta_sigma` (computed as $0.7$ when the cross-layer coefficient of variation of $\sigma_{\text{base}} > 0.5$; `lopi_profiler.py` L210). |
| $M_\perp$ | $M_V - \frac{\langle M_V, V_{\text{ctx}}\rangle}{\|V_{\text{ctx}}\|^2} V_{\text{ctx}}$ | `lopi.py` L352–355 | **R-3 empirical evidence (630-cell ablation):** orthogonal projection *increases* neutral-text drift at $\alpha \in [0.25, 2]$ across all 6 GPT-2 cells. This is the opposite of the design intent. Hypothesis: $M_\parallel$ (the projected-out component) is not "redundant" but contains "alignment prior signal" that the frozen model needs to maintain coherent outputs. `orthogonal` is now `False` by default (`lopi.py` L83). |
| $V_{\text{out}}$ | $V_{\text{ctx}} + \gamma_t \cdot w(\ell, t) \cdot M_\perp$ | `lopi.py` L410 | This formula is mathematically correct. Failure modes propagate from the upstream components, not from the sum itself. |

---

## 3. Detailed Derivations

### 3.1 Derivative Gate $\gamma_t$

**Motivation.** If the topic is stable (similar successive queries), injecting bank memory may be redundant or disruptive. A gate that silences injection when $\Delta_Q$ is small and opens it during topic shifts is a natural safeguard.

**Formula.**

$$\gamma_t = \sigma\!\left(k \cdot (\|Q_t - Q_{t-1}\|_2 - \theta)\right), \quad k = 5,\; \theta = 0.5.$$

When $\|Q_t - Q_{t-1}\| \gg \theta$: $\gamma_t \to 1$ (full injection). When $\|Q_t - Q_{t-1}\| \ll \theta$: $\gamma_t \to 0$ (silent). The sigmoid slope $k = 5$ gives a transition width of $\approx 1/k = 0.2$ in $\Delta_Q$ units.

**Implementation:** `derivative_gate` in `lopi.py` lines 229–247. The function returns a tensor of shape $(B, H, T, 1)$ to broadcast over the head dimension $D$.

**Issue:** The session-boundary fallback (`q_prev is None`) returns $\gamma_t = 1$ (L239–241), which is correct: at the start of a session the first injected step should not be silenced. Shape mismatch (different-length prompts between steps) also triggers fallback.

### 3.2 Depth Signal $\hat{d}_t$ and Layer Gaussian $w(\ell, t)$

**Motivation.** Not all transformer layers are equally receptive to injected knowledge. Mid-to-late layers tend to process semantic content; early layers focus on syntactic processing. A Gaussian $w(\ell, t) = \exp\!\left(-(\ell - \mu_t)^2 / 2\sigma_t^2\right)$ with a dynamically shifted centre $\mu_t$ concentrates injection energy where it is most likely to be useful.

**Static-mode formula ($\texttt{profile\_mode="static"}$):**

$$\hat{d}_t = \sigma\!\left(\kappa_d \left(\frac{\bar{N}_{t-1}}{\texttt{norm\_base}} - 1\right)\right), \quad \kappa_d = 2.0,\; \texttt{norm\_base} = 10.0,$$

$$\mu_t = L \cdot (0.3 + 0.5 \hat{d}_t), \quad \sigma_t = \frac{L}{6} e^{-\beta_\sigma \bar{\sigma}_{\max}}, \quad \beta_\sigma = 2.0.$$

$\bar{N}_{t-1}$ is the mean residual L2 norm across layers from the previous step (causality fix: reads `state.prev_residual_norms`, not the current-step values; see B1 fix documented in `LOPIState` lines 155–166).

**Auto-mode formula ($\texttt{profile\_mode="auto"}$, Phase S):**

$$\hat{d}_t = \sigma\!\left(\kappa_d \cdot \frac{1}{L}\sum_\ell \operatorname{clip}\!\left(\frac{N_t(\ell) - \mu_{\text{base}}(\ell)}{\sigma_{\text{base}}(\ell) + \varepsilon},\; -z_c, z_c\right)\right),$$

$$\mu_t = \mu_{\text{arch}} + c \cdot (\hat{d}_t - 0.5) \cdot L, \quad c = 0.2,$$

$$\sigma_t = \max\!\left(\frac{L}{6} \eta_\sigma,\; \sigma_{\text{floor}}\right), \quad \eta_\sigma \in \{0.7, 1.0\}.$$

Auto mode is implemented in `lopi.py` lines 313–341 (`layer_gaussian_weight`), with the Z-score signal in `_z_depth_signal` lines 250–288.

### 3.3 Orthogonal Novelty $M_\perp$

**Motivation.** The bank value $M_V$ may be largely parallel to the native context value $V_{\text{ctx}}$, in which case injecting it adds no new information. Projecting out the parallel component leaves only the "novel" orthogonal part.

**Formula.**

$$M_\perp = M_V - \frac{\langle M_V, V_{\text{ctx}}\rangle}{\|V_{\text{ctx}}\|^2 + \varepsilon} V_{\text{ctx}}.$$

**Implementation:** `orthogonal_novelty` in `lopi.py` lines 344–355. The epsilon guard handles the zero-$V_{\text{ctx}}$ edge case: when the context value is zero, $M_\perp = M_V$ (no projection).

**Empirical failure (R-3).** The 630-cell ablation grid showed that enabling `orthogonal=True` *increases* neutral-text NLL drift at $\alpha \in [0.25, 2]$ for all 6 GPT-2 configurations tested. Only at the extreme $\alpha = 8$ catastrophic regime does the pure Gaussian+derivative variant (A4) dramatically collapse drift (65–95% reduction). The orthogonal component was therefore disabled by default after R-3 (`lopi.py` line 83).

---

## 4. Causality Fix (Phase S, B1)

The v3.4 implementation read `prev_residual_norms[layer_idx]` at every layer of step $t$ and simultaneously wrote the same dict, meaning layer $\ell + 1$ at step $t$ read the *current step's* norm of layer $\ell$ — violating the "use only $t-1$ information" invariant. Phase S introduced two dicts (`prev_residual_norms` frozen at step start; `pending_residual_norms` written during the step) and a `commit_step()` promotion called between forwards. See `LOPIState` lines 155–222 and `attn_native_bank.py` lines 548–562.

---

## 5. Known Failure Modes (Phase W.2 Must Answer)

**Failure 1: $M_\perp$ drift reversal.** At production $\alpha$ (1–5), the orthogonal projection removes useful signal, causing drift to *increase* vs the no-ortho baseline. Root cause unknown; leading hypothesis is that $M_\parallel$ encodes "alignment prior" information the model needs for coherent generation. Evidence: R-3 630-cell empirical ablation. Phase W.2-Q1 diagnostic: plot $\|M_\perp\|^2 / \|M_V\|^2$ vs per-cell drift; a positive correlation would confirm the hypothesis.

**Failure 2: Dead $\bar{\sigma}_{\max}$ in static $\sigma_t$.** In all production inference paths, `LOPIState.mhc_sigma_max_running` is initialised to 0.0 and never updated (B3). The static-mode $\sigma_t = (L/6) e^{-\beta_\sigma \cdot 0} = L/6$ is therefore a constant, not a dynamic width. The auto-mode fix ($\eta_\sigma$ from profile CV) is correct but only active when `profile_mode="auto"`. Evidence: code inspection (`LOPIState` L189, `update_mhc_sigma` L200–204 is never called by the patcher).

**Failure 3: Gemma-only constants.** Three hyperparameters are calibrated on Gemma-4-E2B:
- $k = 5$, $\theta = 0.5$ (derivative gate, `LOPIConfig` L107–108)
- $\texttt{norm\_base} = 10.0$ (static depth signal, `LOPIConfig` L113)

These numbers are wrong for all non-Gemma architectures. Q norms in Qwen3 are ~5× larger; using $\theta = 0.5$ means $\gamma_t \approx 1$ always (gate is stuck open). `norm_base = 10.0` for a Qwen2.5-0.5B model whose typical residual norm is 2–3 means $\hat{d}_t \approx \sigma(-2\kappa_d) \approx 0.02$ always (depth signal stuck at "shallow"). Both failures make LOPI largely a no-op on non-Gemma models in static mode.

---

## 6. Code Reference

| Component | File | Lines |
|---|---|---|
| `LOPIConfig` dataclass | `lopi.py` | 66–139 |
| `LOPIState` dataclass (causality fix) | `lopi.py` | 146–222 |
| `derivative_gate` ($\gamma_t$) | `lopi.py` | 229–247 |
| `_z_depth_signal` (auto $\hat{d}_t$) | `lopi.py` | 250–288 |
| `layer_gaussian_weight` ($\mu_t$, $\sigma_t$, $w$) | `lopi.py` | 291–341 |
| `orthogonal_novelty` ($M_\perp$) | `lopi.py` | 344–355 |
| `apply_lopi` (top-level, $V_{\text{out}}$) | `lopi.py` | 362–417 |
| Dead $\bar{\sigma}_{\max}$ field | `lopi.py` | 187–190 |
| `orthogonal=False` default (post R-3 update) | `lopi.py` | 83 |
| `LOPIProfile` dataclass | `lopi_profiler.py` | 46–74 |
| `profile_residuals` (one-shot profiling) | `lopi_profiler.py` | 124–229 |
| `_argmax_low_tiebreak` ($\mu_{\text{arch}}$) | `lopi_profiler.py` | 105–110 |
| $\eta_\sigma$ computation | `lopi_profiler.py` | 209–210 |
| LOPI call in patched forward | `attn_native_bank.py` | 536–562 |

---

## 7. Conclusion

LOPI's mathematical framework is sound: the three-component design ($\gamma_t$, $w(\ell, t)$, $M_\perp$) addresses real problems in uncontrolled bank injection. However the current implementation has three confirmed failure modes that must be resolved before LOPI can be claimed production-ready. Phase W.2 is the designated proving ground.

---

*References:* Meng et al., *ROME*, NeurIPS 2022; DeltaMemory PREREGISTRATION lopi\_v33, 2026-05-04; Phase R ablation report `reports/cleanroom/lopi_v33/FINDINGS.md`.
