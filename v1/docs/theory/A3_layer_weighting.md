---
audit_item: A3
verdict: partial
evidence_path: experiments/A3_layer_weighting/raw_cells.json
---

# A3 — Gaussian layer weighting audit

## Diagnosis

`LOPIConfig.profile_mode` defaults to `"auto"`, not a pure static `(L/2,L/6)` fallback. In `layer_gaussian_weight`:

- auto mode uses `profile.mu_arch = argmax_l sigma_base(l)` and `sigma_t = (L/6) * eta_sigma`;
- static mode falls back to depth-driven `mu_t = L*(mu_low + mu_span*sigmoid(...))` and `sigma_t ≈ L/6`.

So the complaint is correct for the fallback intuition, but incomplete for v0.4: the active path is profiler-driven when a profile is attached.

## Why mid/transition layers are plausible V-injection points

The mid-layer prior is not arbitrary, but it is a heuristic:

- **Information bottleneck view:** early layers encode lexical/local features; late layers are close to logits and more brittle. A low-rank external V perturbation is safest where residual features are abstract enough to bind facts but not yet committed to output tokens.
- **Geva et al. 2021, Transformer FFN as key-value memory:** factual associations emerge in intermediate MLP/attention computations, supporting mid-stack edits/readouts.
- **Tenney et al. 2019, BERT pipeline:** syntax/semantic abstractions progress across layers; relation-level features tend to concentrate after early lexical layers and before final task heads.

This supports a middle/transition window but does not derive `L/2,L/6`.

## Two derivable alternatives

### 1. Residual spectral-radius / max-rank rule

Let `H_l` be residual states at layer `l`. Define `σ_max(l)` as the largest singular value of centered `H_l` across tokens/prompts, or a cheap profiler proxy `sigma_base(l)`. Choose:

`l* = argmax_l σ_max(l)`.

Interpretation: inject where the residual stream has maximum local rank/variance and can absorb a rank-N bank expansion with minimum saturation.

Code sketch:

```python
sigmas = profile.sigma_base
mu_arch = argmax_low_tiebreak(sigmas)
cfg.profile_mode = "auto"
state.profile.mu_arch = mu_arch
```

### 2. Logit-lens information-gain rule

Let `P_l(y|x)` be the logit-lens distribution from hidden state `H_l`. Define information gain:

`IG(l)=KL(P_l || P_{l-1})` or `Δ log p(target)` on held-out text.

Choose the peak positive transition:

`l* = argmax_l IG(l)`.

This selects the layer where hidden states most change model predictions. It requires logits or an unembedding lens; the current comparison script includes only a residual-norm proxy when logits are absent.

## Empirical comparison on W-T3.6 profiles

`experiments/A3_layer_weighting/compare_layer_rules.py` recomputes layer rules from existing Qwen profiler artifacts.

| model | L | static peak | profile `mu_arch` | sigma-radius argmax | logit-lens proxy |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 24 | 12 | 5 | 5 | 23 |
| Qwen2.5-1.5B | 28 | 14 | 5 | 5 | 2 |

The profiler consistently selects layer 5, far shallower than `L/2`. The residual logit-lens proxy is unstable and should not be treated as evidence until a true unembedding logit-lens run is added.

## 修复方案

Do not replace the default yet. Keep `profile_mode="auto"` as an ablation flag and document that `(L/2,L/6)` is only a fallback prior. Promote the max-`sigma_base` rule as the current empirical default; require a true logit-lens experiment before making claims about information-gain optimality.
