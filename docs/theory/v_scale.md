# V-Scale (R-7): Bank Value Magnitude Calibration

**Phase U.4** — DeltaMemory theory documentation.

---

## 1. The Problem V-Scale Solves

DeltaMemory injects bank values via

$$\text{out}_{\text{bank}} = W_{:,T:} \cdot (\alpha M_V),$$

and the scalar $\alpha$ is meant to be a **linear** control of injection strength. For $\alpha$ to have comparable meaning across different LLM architectures, the bank values $M_V$ must have comparable magnitudes.

They do not. Empirically:

| Architecture | Native V norm (per head) | Has native `v_norm` | Typical $\|M_V\|$ relative to Gemma |
|---|---|---|---|
| Gemma-4-E2B | Bounded by internal `v_norm` |  | 1× (baseline) |
| Qwen2.5-0.5B | Unbounded |  | ~5–10× |
| Llama-3.2-1B | Unbounded |  | ~3–7× |
| GLM-4-9B | Unbounded |  | ~2–5× |

With $\|M_V\|$ varying by 10×, a fixed $\alpha = 1.0$ represents wildly different injection energies. The V-scale module (Phase R-7) addresses this at **write time**, when bank values are captured, by capping each bank V vector's per-head RMS to a fixed target.

---

## 2. Cap vs. Normalise: Why RMS = 0.5 is a Cap

### 2.1 The distinction

**Normalisation** (unit-RMS): $M_V \leftarrow M_V \cdot (R / \text{RMS}(M_V))$ unconditionally, mapping every V to exactly RMS $= R$. This *amplifies* V vectors that were already small and *shrinks* V vectors that were large.

**Cap** (RMS cap): $M_V \leftarrow M_V \cdot \min(1,\; R / \text{RMS}(M_V))$, equivalent to applying the scale only when $\text{RMS}(M_V) > R$. Small V vectors are passed through unchanged.

The production mode is `auto_rms_cap` with `value_target_rms = 0.5`. The cap logic is:

$$\text{scale} = \frac{R}{\text{RMS}(M_V)}, \quad \text{scale}_{\text{cap}} = \min(1,\; \text{scale}), \quad \tilde{M}_V = M_V \cdot \text{scale}_{\text{cap}}.$$

The `min(1, ·)` clamp is the critical line: it makes the operation a **cap not a normaliser**.

**Implementation:** `_scale_bank_value_capture` in `attn_native_bank.py` lines 77–116. The cap vs. unit-RMS distinction is controlled by `mode in cap_modes` (line 111–112).

### 2.2 Why capping is safer than normalising

A no-`v_norm` model may produce bank V vectors with RMS $\ll 0.5$ on some fact prompts (e.g. rare tokens, short prompts). If we normalise unconditionally, we amplify these small vectors to RMS = 0.5, potentially injecting noise at high energy. The cap preserves small V as-is, avoiding amplification of what may be a low-confidence or degenerate capture.

**Empirical evidence.** In the R-7 regression on Qwen2.5-0.5B, switching from `unit_rms` (unconditional normalise) to `rms_cap` (cap only) changed mean NLL drift from +12.32 nats to a small positive number. The +12.32 nats regression with `unit_rms` is the direct evidence that amplifying small V vectors is harmful. This result is cited in the module docstring and motivated the default `auto_rms_cap` setting.

---

## 3. Why Gemma-4 Does Not Need V-Scale

Gemma-4 attention (`Gemma3nTextAttention`) applies a learnable `v_norm` layer to the V projection output before the softmax-weighted sum. This layer was trained to keep V activations at a controlled scale. When `auto_rms_cap` is active and `has_native_v_norm = True`, the function returns immediately without any scaling:

```python
if mode == "none" or (mode in auto_modes and has_native_v_norm):
    return v
```

(`attn_native_bank.py` line 97–98.)

The `has_native_v_norm` flag is queried from the `ArchAdapter` for each attention module via `adapter.has_native_v_norm(self)` (`attn_native_bank.py` line 427). For Gemma-4, this returns `True`; for Qwen/Llama/GLM it returns `False`.

This auto-dispatch ensures that Gemma-4's already-normalised V vectors are not double-capped, while no-`v_norm` families get the cap applied at capture time.

---

## 4. Energy vs. Direction: Why V-Scale Alone Is Not Enough

The V-scale cap bounds the **energy** (magnitude) of injected bank values:

$$\|\text{out}_{\text{bank}}\|_F \leq \sigma_{\max}(W_{:,T:}) \cdot \alpha \|\tilde{M}_V\|_F \leq \sigma_{\max}(W_{:,T:}) \cdot \alpha \cdot (R \cdot \sqrt{q}).$$

But it says nothing about the **direction** of $\tilde{M}_V$ relative to the model's residual stream basis. Two bank values with identical RMS but pointing in completely different directions will have entirely different effects on the model's downstream computation.

For $\alpha$ to be **linearly comparable across architectures** (i.e. $\alpha = 1$ means "the same degree of perturbation" regardless of model family), we need:
1. **Energy comparability** — V-scale (R-7) provides this.
2. **Direction comparability** — the bank value must encode the "same amount of semantic content" per unit of head-dimension. This is the problem that Phase W.4 (activation-steering direction alignment, e.g. CAA / RepE) is designed to address.

V-scale is the first half of the solution. Without the second half, $\alpha = 1$ on Gemma means something different from $\alpha = 1$ on Qwen even after capping — the cap sets a ceiling on magnitude but cannot enforce semantic equivalence of direction. Phase W.4 tests whether CAA-style steering vectors provide the missing direction constraint.

---

## 5. Interaction with mHC

mHC (column-cap) bounds $\sigma_{\max}(W_{:,T:}) \leq \kappa$ by constraining the routing matrix. V-scale bounds $\|\tilde{M}_V\|_F \leq R \sqrt{q}$ by constraining the value magnitude. Together they give:

$$\|\text{out}_{\text{bank}}\|_F \leq \kappa \cdot \alpha \cdot R \sqrt{q}.$$

This is a **joint upper bound** on injection energy that is linear in $\alpha$ and independent of architecture (given both caps are active). Neither constraint alone is sufficient: mHC alone allows large $\|M_V\|$ to cause blow-up; V-scale alone allows large $\sigma_{\max}(W_{:,T:})$ to cause routing-mediated blow-up. They are two sides of the same energy inequality and must be used together.

---

## 6. Code Reference

| Component | File | Lines |
|---|---|---|
| `ValueScaleMode` type alias | `attn_native_bank.py` | 74 |
| `_scale_bank_value_capture` (full implementation) | `attn_native_bank.py` | 77–116 |
| Cap vs. unit-RMS branch (`clamp max=1`) | `attn_native_bank.py` | 111–112 |
| Auto-mode Gemma bypass (`has_native_v_norm`) | `attn_native_bank.py` | 97–98 |
| `value_scale_mode`, `value_target_rms` on bank | `attn_native_bank.py` | 163–167 |
| V-scale call at capture time | `attn_native_bank.py` | 419–431 |
| `has_native_v_norm` query per layer | `attn_native_bank.py` | 427 |

---

## 7. Summary

R-7 V-scale is a write-time operation that caps the per-head RMS of captured bank V vectors to 0.5 for model families without native `v_norm` (Qwen/Llama/GLM), while leaving Gemma-4 untouched. The use of a cap (not normalisation) is deliberate: amplifying small V vectors was shown to cause +12.32 nats NLL regression on Qwen2.5-0.5B. V-scale addresses injection energy; the orthogonal problem of injection direction comparability remains open and is the subject of Phase W.4.

---

*References:* DeltaMemory Stage R-7 implementation notes; Phase Q2 sweep results (168-cell drift grid); `attn_native_bank.py` module docstring.
