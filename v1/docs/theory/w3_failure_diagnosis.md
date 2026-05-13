# W.3/W.4 failure diagnosis: why CAA joined mHC and LOPI on the wrong side of the gate

W.3 is now closed with an uncomfortable but useful result: all three baselines failed the v0.4 methodology gate. The gate required at least a 30% drift reduction against the `none` baseline at production scale (`alpha >= 1`) under paired tests. The W.4 CAA aggregate contains 5,041 raw cells, 5,040 usable cells, 42 paired Wilcoxon tests, Holm-Bonferroni correction, and zero alpha-zero redline violations (`experiments/W4_caa_baseline/REPORT.md:3-7`). The final W.3 addendum records the same picture: 5,041 cells over three loaded dense models, with gemma cells still queued but unable to reverse the observed direction (`experiments/W3_decision/DECISION.md:185-190`, `experiments/W3_decision/DECISION.md:264-270`).

The headline is not merely "CAA did not win." It is that CAA was usually significant in the wrong direction. For alpha >= 1, all 12 CAA-vs-none tests are Holm-significant at p < 0.01, but 11/12 have positive `median_diff = drift_caa - drift_none`, meaning CAA increases neutral NLL drift (`experiments/W4_caa_baseline/REPORT.md:71-86`). The sole beneficial production-alpha cell is Qwen2.5-0.5B at alpha=1, median_diff = -0.4888; W.3 converts this to about 19% reduction relative to the mean none drift 2.594, below the 30% bar (`experiments/W3_decision/DECISION.md:212-228`). GPT-2-medium is especially diagnostic: its `none` path has drift exactly 0 at every alpha, while CAA adds 0.333, 2.352, 5.594, and 6.073 nats at alpha 1, 2, 4, 8 (`experiments/W4_caa_baseline/REPORT.md:55-68`).

## Code facts that constrain the diagnosis

CAA is implemented as standard additive activation steering: `s = mean_pos(h_l) - mean_neg(h_l)` (`deltamemory/memory/caa_injector.py:218-223`, `deltamemory/memory/caa_injector.py:295-299`) and the hook returns `hidden + alpha * gamma * s` (`deltamemory/memory/caa_injector.py:323-355`). W.4 froze the vanilla path: `inject_layer="mu_arch"`, `use_lopi_gate=False`, and one CAA injector calibrated once per model (`experiments/W4_caa_baseline/PREREG.md:51-58`, `experiments/W4_caa_baseline/run.py:438-450`). The CAA runner explicitly disables the bank for CAA: the method description says `caa` is residual-stream CAA and "bank disabled" (`experiments/W4_caa_baseline/run.py:6-10`).

Layer choice is the weak link. `_resolve_layer()` uses the LOPI profiler when available and falls back to `L // 2` (`deltamemory/memory/caa_injector.py:143-176`). The profiler chooses `mu_arch = argmax_l sigma_base(l)`, i.e. the layer with largest residual-norm standard deviation, not the layer with best causal effect on the target metric (`deltamemory/memory/lopi_profiler.py:8-14`, `deltamemory/memory/lopi_profiler.py:225-242`). LOPI's Gaussian path similarly uses `mu_t = profile.mu_arch + c(d_t - 0.5)L` and `sigma_t = (L/6) eta_sigma` in auto mode (`deltamemory/memory/lopi.py:327-348`), with a legacy static fallback equivalent to a broad mid-stack prior (`deltamemory/memory/lopi.py:349-367`). The A3 audit already found that this prior is only heuristic and that Qwen profiles selected layer 5 rather than the static mid-layer 12/14 (`docs/theory/A3_layer_weighting.md:11-17`, `docs/theory/A3_layer_weighting.md:63-72`).

The V-scale cap is real but not a CAA explanation. Bank values are RMS-capped at 0.5 for no-v-norm families (`deltamemory/memory/attn_native_bank.py:77-116`, `deltamemory/memory/attn_native_bank.py:161-166`), and the alpha-zero bit-equality redline is enforced by short-circuiting injection when `alpha <= 0`/empty bank (`deltamemory/memory/attn_native_bank.py:490-497`). But CAA in W.4 does not read the bank, so H_M3 cannot directly generate the 11/12 CAA sign reversal.

## Hypothesis analysis

### H_M1: projection direction inverted

If the only bug were the sign of `pos - neg`, the first-order NLL change at layer l would be

`Delta L(alpha) = alpha * grad_l^T s + O(alpha^2)`.

Replacing `s` with `-s` flips the linear term. A pure sign bug therefore predicts an approximately antisymmetric result at small alpha: the 11 harmful cells should become wins, while the one Qwen2.5-0.5B alpha=1 win should become harmful. This is superficially attractive because the observed pattern is almost globally sign-reversed.

However, code and literature both argue against H_M1 as the primary diagnosis. The implementation uses the conventional CAA/ActAdd direction, positive minus negative, and the W.4 calibration labels are named `positive` and `neutral` in that order (`experiments/W4_caa_baseline/run.py:106-117`). Rimsky et al. (2024) CAA, Li et al. (2023) ITI, and Zou et al. (2023) RepE all use contrastive directions whose sign is task-defined; the risk is not that subtraction is universally backward, but that the chosen concept direction is not aligned with our drift objective. More importantly, GPT-2-medium has zero `none` drift, so any nonzero off-manifold residual shift increases measured drift regardless of semantic sign. H_M1 predicts a useful ablation, `alpha < 0`, but not the full 11/12 pattern by itself.

### H_M2: layer weighting/selection lacks metric theory

This is the most likely root cause. CAA steering is layer-sensitive in the activation-steering literature: ITI selects heads by probe accuracy, RepE uses PCA directions at layers where concepts are linearly separated, and CAA reports middle-layer optima rather than a universal layer. Mneme currently selects a single layer from residual variance (`sigma_base`) or falls back to a broad architectural prior. Neither estimates the causal derivative of neutral NLL drift.

Let `J_l` be the downstream Jacobian from residual activation `a_l` to logits, and `g_l = J_l^T grad_logits L`. CAA injects `alpha s_l`, so first order drift is `alpha g_l^T s_l`; the second-order damage is approximately `(alpha^2/2) s_l^T H_l s_l`. A variance peak maximizes available activation energy, not negative `g_l^T s_l` or low curvature. It can therefore consistently choose a high-gain layer where the steering vector is legible to the model but legible in the wrong loss direction. That is exactly the 11/12 pattern: strong Holm significance, mostly wrong sign, and worse damage as alpha grows.

A measured per-layer Delta-NLL rank fixes the target mismatch. Calibrate `s_l` at candidate layers, run a small neutral/counter-prior validation panel, and select layers with negative paired drift or favorable Pareto slope. This would not guarantee a pass, but it directly optimizes the term whose sign failed.

### H_M3: RMS cap over-clips at alpha >= 1

H_M3 predicts failures in bank-based `none` and LOPI arms, especially on Qwen/Llama/GLM families without native v-norm. It predicts little or no effect on CAA because W.4's CAA arm bypasses bank K/V injection. Therefore it cannot explain the central 11/12 CAA sign reversal. It remains relevant for why raw bank and LOPI failed earlier, but it is not the CAA root cause.

## Verdict

Primary root cause: H_M2, metric-blind layer selection/weighting. Secondary risk: H_M1 should be tested with a signed-alpha ablation, but the standard direction and GPT-2 zero-baseline behavior make it less likely as a universal implementation bug. H_M3 is rejected for CAA because the CAA W.4 path is residual-stream-only. The next method should keep alpha-zero identity but replace single-vector, single-layer additive steering with a measured, low-rank, non-expansive contrastive subspace selected by per-layer Delta-NLL ranks.
