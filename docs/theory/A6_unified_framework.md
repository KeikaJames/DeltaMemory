---
audit_item: A6
verdict: resolved
evidence_path: docs/theory/A6_unified_framework.md
---

# A6 — Unified theoretical framework for Mneme v0.4

## Minimal core

### P1 — Frozen LLM as a functional plus rank-N attention expansion

A frozen LLM is a functional `f_θ` composed of attention and FFN blocks. Mneme does not train `θ`. External memory augments the attention input space by adding `N` K/V slots:

`Attn(Q,K,V) -> Attn(Q, [K; M_K], [V; α M_V])`.

This is a rank-N expansion of the attention readout domain, not a parameter update.

### P2 — α=0 bit equality as degeneracy

Let `I_α` be an injection operator and `F_α = f_θ ∘ I_α`. The red line is:

`F_0(x,M) = f_θ(x)` bit-for-bit.

A sufficient differentiable condition is `I_0 = identity` and the injected branch is multiplied by α, so the perturbation vanishes at α=0. In Fréchet terms, the memory-dependent component has zero value at α=0; implementation additionally requires the branch not to perturb native arithmetic when α=0.

## Theorems

### T1 — Zero-training injection normal form

Under P1+P2, any smooth zero-training injection that only changes value readout can be expanded near α=0 as:

`V_out = V_ctx + α · g(M_V, V_ctx, layer_state) + O(α²)`.

`g` is the first-order perturbation direction. Examples:

- additive bank: `g = W_bank M_V`;
- LOPI: `g = γ_t w_l (M_V - proj_{V_ctx} M_V)`;
- CAA/activation steering: `g` is a residual-space steering vector;
- ECOR: first-order term is rotation direction toward `M_perp`; norm preservation appears in higher-order terms.

### T2 — Orthogonal perturbation gives V-scale calibration

If `<g,V_ctx>=0`, then:

`||V_ctx + αg||² = ||V_ctx||² + 2α<V_ctx,g> + α²||g||² = ||V_ctx||² + α²||g||²`.

Thus first-order energy drift vanishes. V-scale calibration follows: bound `||g||` so α has comparable meaning across architectures. Without orthogonality, a linear cross-term can dominate and α is not an architecture-independent knob.

### T3 — Open: mHC column cap and bank softmax spectral norm

For post-softmax bank columns `W_B`, mHC caps column sums. A target theorem is:

`σ_max(W_B) <= κ` under bank-column cap and nonnegative row-substochastic weights.

The code-level bound is plausible via matrix norm inequalities, but exact tightness with sink mass and mixed native/bank softmax is open. Treat as a safety bound, not a complete theorem.

## Mechanism map

| mechanism | framework role |
|---|---|
| W.1 mHC | mask/weight shaping on attention rows/columns; constrains `W_B` in `g=W_B M_V` |
| W.2 LOPI ablation | tests orthogonal `g`; empirical FAIL/weak result means orthogonality alone is not sufficient |
| W.3 V-scale | constrains `||M_V||` and therefore `||g||` |
| W.4 CAA / RepE | chooses `g` in residual semantic direction rather than native bank V direction |
| W.5 U-LOPI profiler | chooses layer weights `w_l`; evidence says stable layer argmax but mixed downstream quality |
| W.6 α sweeps | estimates perturbation linearity and safe operating interval |
| W.7 long context | tests whether rank-N expansion survives large T softmax dilution |
| W.8 multi-fact | tests interference among columns of `M_K/M_V` |
| W.9 multi-turn | tests state reset and time-varying `M` under repeated writes |
| W.10 diagnostics | measures internal `W_B`, `γ_t`, `w_l`, residual norms to validate `g` assumptions |
| W.12 persistence | preserves `M` and calibration without adding trainable θ |
| W.13 arch adapters | preserve P1 across RoPE/GQA/v_norm conventions without touching W_q/W_k/W_v/W_o |
| W.14 capture policy | chooses which write-token state becomes a row of `M_K/M_V` |

## Incorporating the W.2 ablation FAIL

The orthogonal-novelty corollary “dropping the parallel component should reduce drift while preserving recall” was empirically weakened/denied in W.2/W-T3 notes: orthogonal projection increased neutral-text drift in some regimes, while Gaussian focusing alone was safer. In this framework, the failed corollary is:

`<g,V_ctx>=0 => better task behavior`.

T2 only proves an energy identity. It does **not** prove semantic alignment, recall, or NLL improvement. The main plan must therefore separate:

- mathematically justified energy control: keep V-scale and α=0 degeneracy;
- empirically weak semantic claims: demote LOPI orthogonality and U-LOPI auto to ablation flags;
- still-promising safety controls: retain mHC as an open spectral bound plus diagnostics.

## Red-line compatibility

All mechanisms in the framework must satisfy:

1. no new `nn.Parameter`;
2. no mutation of W_q/W_k/W_v/W_o;
3. α=0 path equals the frozen baseline bit-for-bit;
4. any non-default theoretical experiment must be flag-gated.

Any hand-wave above is explicitly marked open; the rest is a local perturbation theory, not a global guarantee of factual editing quality.
