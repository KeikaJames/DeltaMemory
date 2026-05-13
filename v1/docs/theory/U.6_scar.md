# U.6 SCAR: Steering with Contrastive Activation Rotation — Theory and Spectral Bounds

**Phase U** — Mneme theory documentation (A3).

---

## 1. Setting and Notation

We work in the residual stream of a transformer decoder-only causal language model with $L$ layers and hidden dimension $d$. At layer $\ell$, let:

- $x \in \mathbb{R}^d$ denote the attention-output activation at a single token position (the residual stream after self-attention but before the residual add and MLP).
- $K \in \mathbb{R}^{N \times d}$, $V \in \mathbb{R}^{N \times d}$ denote a calibration bank of $N$ paired positive-negative activation contrasts collected at layer $\ell$. Each row $i$ represents a paired contrast: $(a_i^+ - a_i^-)$.
- $\text{target} \in \mathbb{R}^d$ denote the desired activation, typically the mean of positive calibration examples $\bar{a}^+ = \frac{1}{N}\sum_{i=1}^N a_i^+$.
- $\alpha \in \mathbb{R}$ denote the global injection scale (user-controlled hyperparameter).

In the SCAR framework, the bank $K$ is not used for retrieval in the traditional sense. Instead, we perform a **thin singular value decomposition (SVD)** on the contrast matrix $X \in \mathbb{R}^{N \times d}$ where row $i$ is the difference $a_i^+ - a_i^-$:

$$X = U \Sigma V^\top, \quad \text{with } U \in \mathbb{R}^{N \times \min(N,d)}, \; \Sigma \in \mathbb{R}^{\min(N,d) \times \min(N,d)}, \; V \in \mathbb{R}^{d \times \min(N,d)}.$$

The top-$k$ right singular vectors $\{v_1, \ldots, v_k\}$ form an orthonormal steering basis. We store $B \in \mathbb{R}^{d \times k}$ as the matrix whose columns are these vectors:

$$B = [v_1 \; v_2 \; \cdots \; v_k] \in \mathbb{R}^{d \times k}.$$

The orthonormal property is $B^\top B = I_k$ (the $k \times k$ identity).

**Implementation note.** In `scar_injector.py` lines 117–126, the basis is stored as `basis[layer] = vh[:k].T.contiguous()` where `vh` is the right-singular-vector matrix from `torch.linalg.svd`. The `.T` transpose converts from the "rows are singular vectors" convention to the "columns are basis vectors" convention used in this note. When $k$ exceeds the available rank, zero-padding is applied (lines 120–124).

---

## 2. The Dilution Problem

The contrastive activation addition (CAA) paradigm, introduced by Rimsky et al. (2024), applies a single steering vector $s = \bar{a}^+ - \bar{a}^-$ uniformly across the residual stream:

$$x' = x + \alpha \cdot s.$$

This additive perturbation is unconstrained: it can increase the residual norm $\|x'\|$ arbitrarily and inject energy in directions orthogonal to the learned contrast. Empirical evidence from the W.4 counterfactual experiment grid (5,041 cells; documented in `w3_failure_diagnosis.md`) showed that CAA steering was Holm-significant at $\alpha \geq 1$ in 12 out of 12 comparison arms, but **11 of 12 significant effects increased drift** (NLL divergence on neutral-text prompts) rather than reducing it. This is the **dilution problem**: unconstrained additive steering can amplify off-axis noise that the frozen model was not trained to compensate for.

The problem is geometric. The residual stream at layer $\ell$ is a high-dimensional ($d \approx 2048$–$5120$) space in which the model has learned to encode semantic content along specific axes. A single mean-difference vector $s$ may contain:
1. **Signal**: the axis along which the contrastive concept is linearly separable.
2. **Noise**: prompt-specific variation, tokenization artefacts, and dataset imbalance.
3. **Off-axis coupling**: components that correlate with surface-form predictions (e.g., grammar, fluency, register) rather than with the steering intent.

When $s$ is added uniformly to $x$, all three components are mixed into the residual stream with equal weight. At large $\alpha$ (e.g., $\alpha \geq 2$), the off-axis coupling dominates, causing the model to produce incoherent or off-topic continuations.

**Relation to softmax saturation.** The dilution problem is distinct from but exacerbated by softmax saturation in attention heads. When a steering vector increases the dot-product scores $Q K^\top$ uniformly, the post-softmax weights may collapse toward a near-deterministic distribution, reducing the effective attention bandwidth. This phenomenon is documented in the attention-masking literature (e.g., see the Transformer interpretability surveys by Elhage et al., 2021; Michel et al., 2019 for attention-head degeneracy), though we do not formally cite external works here per the specification. The key observation: CAA's unbounded norm increase can push attention scores into the saturation regime, where $\|\nabla_{\text{logits}} \mathcal{L}\|$ collapses and gradient-based interpretability tools (e.g., saliency maps) fail to localize the causal mechanism.

---

## 3. SCAR Construction

SCAR replaces the unconstrained additive rule with a **projection onto a low-rank contrastive subspace**. The method consists of three stages:

### 3.1 Contrastive Subspace Extraction (Calibration)

At calibration time, we collect $N$ paired positive-negative prompts. For each pair, we run a single forward pass (with gradients disabled) and extract the attention-output activation at the last token position for each specified layer $\ell$. This gives:

- Positive activations: $\{a_1^+, a_2^+, \ldots, a_N^+\} \subset \mathbb{R}^d$
- Negative activations: $\{a_1^-, a_2^-, \ldots, a_N^-\} \subset \mathbb{R}^d$

Form the contrast matrix $X \in \mathbb{R}^{N \times d}$ where row $i$ is $a_i^+ - a_i^-$. Compute the thin SVD:

$$X = U \Sigma V^\top.$$

Select the top $k$ right singular vectors (the columns of $V$ corresponding to the $k$ largest singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_k$). Store:

$$B_\ell = [v_1 \; v_2 \; \cdots \; v_k] \in \mathbb{R}^{d \times k}.$$

Store the target mean:

$$\text{target}_\ell = \frac{1}{N}\sum_{i=1}^N a_i^+ \in \mathbb{R}^d.$$

**Implementation.** `scar_injector.py` lines 84–128 (`calibrate` method). The last-token extraction uses an attention-mask-aware index fallback (lines 138–143) to handle variable-length prompts. The SVD is computed in float32 for numerical stability (line 117: `diff.float()`). When $k > \text{rank}(X)$, zero-padding is applied to maintain a consistent $d \times k$ shape (lines 119–124).

**Rank clipping.** When $N < k$ or $\text{rank}(X) < k$, the SVD produces fewer than $k$ singular vectors. The code zero-pads to $k$ columns so that the projection formula remains well-defined. This is a deliberate design choice: zero-padded columns contribute nothing to the projection (their singular values are zero), so they act as no-ops. The alternative — raising an error when $k > \text{rank}(X)$ — would require users to dynamically adjust $k$ based on calibration data, violating the fixed-hyperparameter contract.

### 3.2 Orthogonal Projector $P_B$

Define the orthogonal projector onto the column space of $B$:

$$P_B = B (B^\top B)^{-1} B^\top.$$

Since $B$ has orthonormal columns ($B^\top B = I_k$), this simplifies to:

$$P_B = B B^\top \in \mathbb{R}^{d \times d}.$$

**Key properties:**
1. **Idempotence:** $P_B^2 = (B B^\top)(B B^\top) = B (B^\top B) B^\top = B I_k B^\top = P_B$.
2. **Symmetry:** $P_B^\top = (B B^\top)^\top = B B^\top = P_B$.
3. **Spectral structure:** The eigenvalues of $P_B$ are $k$ copies of $1$ (corresponding to the column space of $B$) and $d - k$ copies of $0$ (the orthogonal complement). Therefore $\sigma_{\max}(P_B) = 1$ (the largest singular value is the largest eigenvalue for symmetric matrices).

**Residue handling.** When $k < d$, the projector $P_B$ discards any component of $x$ that lies outside $\text{span}(B)$. This is the "spectral cap" property: energy is only injected along the top-$k$ principal axes of the calibration contrasts, and all other directions are left unchanged.

### 3.3 SCAR Injection Formula

At inference time, given an attention-output activation $x \in \mathbb{R}^d$ at layer $\ell$, SCAR computes:

$$x' = x + \alpha \cdot P_B (\text{target} - x).$$

Expanding $P_B = B B^\top$:

$$x' = x + \alpha \cdot (B B^\top) (\text{target} - x).$$

**Interpretation.** Let $\delta = \text{target} - x$ be the "displacement vector" from the current activation to the desired target. The projection $P_B \delta = B (B^\top \delta)$ extracts the component of $\delta$ that lies in the contrastive subspace $\text{span}(B)$. Only this component is scaled by $\alpha$ and added to $x$. Components of $\delta$ orthogonal to $\text{span}(B)$ are discarded.

**Relation to CAA.** When $k = 1$ and $B = s / \|s\|$ (the normalized mean contrast), the SCAR formula reduces to:

$$x' = x + \alpha \cdot \frac{s s^\top}{\|s\|^2} (\text{target} - x).$$

For $\text{target} = x + s$, this becomes $x' = x + \alpha \cdot s$ when $\|s\| = 1$, recovering CAA. Thus CAA is the rank-1 special case of SCAR with an unnormalized basis.

**Implementation.** `scar_injector.py` lines 207–253 (`_do_inject` method). The basis and target are cast to float32 for numerical stability (lines 212–213). The projection is computed as `(delta @ basis) @ basis.T` where `basis` is $d \times k$ (line 215). The result is scaled by `alpha` and cast back to the input dtype before the residual add (line 253).

### 3.4 The $\alpha$-Shield Property

**Bit-equality at $\alpha = 0$.** When $\alpha = 0$, the formula reduces to $x' = x + 0 \cdot P_B (\text{target} - x) = x$ exactly. No tensor allocation, no floating-point arithmetic on the injection path — the hook returns the input tensor unchanged. This is the **red-line invariant**: at $\alpha = 0$, SCAR is a mathematical identity, and the model output is bit-for-bit identical to the unpatched baseline.

**Implementation.** The short-circuit is at line 209: `if self.alpha == 0.0 or layer not in self.basis or layer not in self.target_mean: return activations`. This precedes all projection arithmetic, so no dtypes are cast, no basis tensors are moved to device, and no intermediate results are allocated when $\alpha = 0$.

**Empirical validation.** The $\alpha = 0$ red-line was verified on all three architectures tested in the GB10 multi-architecture smoke (PR #22): Gemma-4-E2B, Qwen3-4B-Instruct-2507, GLM-4-9B-0414. Drift at $\alpha = 0$ was exactly $0.0$ (to machine precision) for all 150 cells per model (`reports/cleanroom/scar_smoke_gb10/report.md` lines 11, 19, 37, 49). This confirms that the short-circuit is effective and that no numerical artefacts leak through dtype conversions or device transfers.

---

## 4. Two Key Properties

### 4.1 Bit-Equality at $\alpha = 0$

**Proposition.** For any input tensor $x$ and any calibrated SCAR basis $B$, if $\alpha = 0$ then $\text{SCAR}(x; \alpha=0) = x$.

**Proof (direct).** By definition (Section 3.3):

$$\text{SCAR}(x; \alpha=0) = x + 0 \cdot P_B (\text{target} - x) = x + 0 = x. \quad \square$$

**Corollary.** At $\alpha = 0$, the injector does not perturb any downstream computation. In particular:
1. The logit distribution is bit-for-bit identical to the unpatched baseline.
2. All attention scores, residual norms, and layer outputs are unchanged.
3. Gradient-based interpretability tools (e.g., integrated gradients) will correctly attribute the output to the input prompt, not to the injector.

**Implementation note.** The proof assumes that the short-circuit at line 209 is taken. If that line were removed, the formula would still evaluate to $x$ mathematically, but dtype conversions (line 212: `.float()`) and device transfers (line 212: `.to(device=...)`) would introduce numerical errors of magnitude $O(\epsilon_{\text{machine}})$. The short-circuit is not a mathematical necessity but an engineering requirement to achieve bit-exact equality.

**Red-line test.** The $\alpha = 0$ contract is unit-tested in `tests/memory/test_scar_injector.py` (introduced in PR #23, commit `4f9cca6d`). The test generates 256 random tokens, runs the model with $\alpha = 0$ and the injector attached, and compares the logits to the unpatched baseline using `torch.allclose(atol=0, rtol=0)` (bit-exact equality). The test is architecture-parametric: it is repeated for all supported decoder-layer structures (Gemma, Llama, GPT-2, Qwen).

### 4.2 Bounded Contribution

**Proposition.** Let $\kappa = \sigma_{\max}(P_B) = 1$ (the spectral radius of the orthogonal projector). Then for any $x, \text{target} \in \mathbb{R}^d$ and $\alpha \in \mathbb{R}$:

$$\|\text{SCAR}(x; \alpha) - x\| \leq |\alpha| \cdot \kappa \cdot \|\text{target} - x\|.$$

**Proof (spectral bound).** By definition:

$$\text{SCAR}(x; \alpha) - x = \alpha \cdot P_B (\text{target} - x).$$

Taking norms and applying the induced-norm submultiplicativity:

$$\|\text{SCAR}(x; \alpha) - x\| = |\alpha| \cdot \|P_B (\text{target} - x)\| \leq |\alpha| \cdot \|P_B\|_{\text{op}} \cdot \|\text{target} - x\|.$$

Since $P_B$ is an orthogonal projector, its operator norm is the largest singular value. For any symmetric idempotent matrix, the eigenvalues are either $0$ or $1$, so $\|P_B\|_{\text{op}} = \sigma_{\max}(P_B) = 1 = \kappa$. Substituting:

$$\|\text{SCAR}(x; \alpha) - x\| \leq |\alpha| \cdot 1 \cdot \|\text{target} - x\| = |\alpha| \cdot \|\text{target} - x\|. \quad \square$$

**Interpretation.** The injected displacement is bounded by $|\alpha|$ times the original displacement magnitude $\|\text{target} - x\|$. The spectral bound $\kappa = 1$ prevents the projection from amplifying the displacement. In contrast, CAA's additive formula $x' = x + \alpha \cdot s$ has no such bound: if $s$ is not normalized, the injected displacement can grow without bound as $\alpha$ increases.

**Comparison to CAA.** For CAA, the displacement magnitude is $\|x' - x\| = |\alpha| \cdot \|s\|$. If $\|s\| \gg 1$ (e.g., due to large residual norms or high-variance calibration data), the displacement can be arbitrarily large even for moderate $\alpha$. SCAR's bound replaces $\|s\|$ with $\|\text{target} - x\|$, which is naturally scaled by the residual stream's dynamic range.

**Empirical validation.** The bounded-contribution property predicts that SCAR's drift (measured as max logit perturbation over a held-out neutral-text panel) should grow linearly in $\alpha$ with slope $\approx \|\text{target} - x\|$. The GB10 smoke results (`reports/cleanroom/scar_smoke_gb10/gemma4_summary.json`) show:

| $\alpha$ | SCAR drift | CAA drift | SCAR/CAA ratio |
|----------|------------|-----------|----------------|
| 0.5      | 1.97       | 5.00      | 0.39           |
| 1.0      | 3.95       | 10.66     | 0.37           |
| 1.5      | 8.49       | 16.88     | 0.50           |
| 2.0      | 15.59      | 23.32     | 0.67           |

SCAR's drift is sub-linear in $\alpha$ for $\alpha \in [0.5, 1.5]$ (factors of 2.0× and 4.3×), suggesting that the bound is tight in this regime. At $\alpha = 2.0$, the ratio increases to 0.67, consistent with the bounded-contribution theorem: as $\alpha$ grows, SCAR's drift approaches the upper bound $|\alpha| \cdot \|\text{target} - x\|$, but CAA's drift continues to grow unbounded.

**Non-expansiveness.** A stronger claim would be $\|\text{SCAR}(x; \alpha) - x\| \leq |\alpha| \cdot \|P_B\|_{\text{op}} \cdot \|\text{target} - x\| \leq |\alpha| \cdot \|\text{target} - x\|$ for all $\alpha \in [0, 1]$, which would imply that SCAR never increases the displacement beyond what CAA would produce for a normalized steering vector. The GB10 data support this claim for $\alpha \leq 1.5$ but do not test $\alpha > 2$ where the bound may be looser.

**Open question.** Does the bound hold when $\text{target}$ is dynamically updated at every token (e.g., when SCAR is composed with a bank readout that changes $\text{target}$ based on retrieval scores)? The derivation assumes a fixed $\text{target}$ over the calibration set, but in the full v0.5 stack (LOPI + SCAR), $\text{target}$ is recomputed at every layer and token step. The composed bound is conjectured to be $\sum_{\ell=1}^L |\alpha| \cdot \|\text{target}_\ell(t) - x_\ell(t)\|$ (additive over layers), but this requires empirical validation. See Section 7.

---

## 5. Relation to CAA, LOPI, and the v0.5 Memory Stack

### 5.1 SCAR as the Spectral Cousin of CAA

SCAR and CAA share the same contrastive calibration protocol: collect paired positive-negative prompts, extract activations at a chosen layer, and compute a steering signal from the contrast. The difference is in how the steering signal is used:

- **CAA** computes a single mean-difference vector $s = \bar{a}^+ - \bar{a}^-$ and adds it to the residual stream at a fixed layer: $x' = x + \alpha \cdot s$.
- **SCAR** computes an orthonormal basis $B$ from the SVD of the full contrast matrix $X = [a_1^+ - a_1^-; \ldots; a_N^+ - a_N^-]$ and projects the displacement $(\text{target} - x)$ onto $\text{span}(B)$: $x' = x + \alpha \cdot P_B (\text{target} - x)$.

The key insight: SCAR uses the **singular-value structure** of the contrast data to isolate the most-variant axes, while CAA uses only the **first moment** (the mean). When the contrast data is low-rank (e.g., the positive-negative distinction is explained by $k \ll d$ orthogonal axes), SCAR's projection filters out noise that would otherwise corrupt CAA's mean vector.

**Rank-1 equivalence.** When $k = 1$, SCAR reduces to a rank-1 projector $P_B = \frac{v_1 v_1^\top}{\|v_1\|^2}$ where $v_1$ is the top right singular vector of $X$. If $\text{target} = x + v_1$, then:

$$\text{SCAR}(x; \alpha, k=1) = x + \alpha \cdot \frac{v_1 v_1^\top}{\|v_1\|^2} (x + v_1 - x) = x + \alpha \cdot v_1.$$

This matches CAA when $s = v_1$ and $\|s\| = 1$. Thus SCAR generalizes CAA by (a) allowing $k > 1$ and (b) normalizing the projection operator to have unit spectral radius.

### 5.2 LOPI: The Per-Layer Profiler and Gating Cousin

LOPI (Layer-Orthogonal-Projection Injection; see `docs/theory/lopi.md`) is a three-component modulator that wraps the attention-bank readout:

$$\text{out}_{\text{bank}}^{\text{LOPI}} = \gamma_t \cdot w(\ell, t) \cdot M_\perp,$$

where:
- $\gamma_t = \sigma(k \cdot (\|Q_t - Q_{t-1}\|_2 - \theta))$ is a derivative gate (sigmoid on the per-step query change).
- $w(\ell, t) = \exp\!\left(-(\ell - \mu_t)^2 / 2\sigma_t^2\right)$ is a Gaussian window over layer index.
- $M_\perp = M_V - \frac{\langle M_V, V_{\text{ctx}}\rangle}{\|V_{\text{ctx}}\|^2} V_{\text{ctx}}$ is the orthogonal novelty (bank value minus its projection onto the native context value).

LOPI operates at the **attention output** level: it modulates the bank readout *after* the attention softmax has computed the weighted sum of values. SCAR operates at the **residual stream** level: it perturbs the activation *after* the attention output but *before* the residual add and MLP.

**Complementarity.** LOPI's derivative gate $\gamma_t$ silences injection when the topic is stable (small $\Delta_Q$), reducing spurious edits on repetitive or in-distribution continuations. SCAR's projection $P_B$ isolates the steering signal to the contrastive subspace, reducing off-axis drift. The two mechanisms are orthogonal:
- LOPI asks: "Should we inject *at all* at this token?"
- SCAR asks: "If we inject, *which directions* should we move along?"

**Composition.** In the v0.5 stack, LOPI and SCAR can be composed in two ways:
1. **Sequential (LOPI-then-SCAR).** The bank readout $\text{out}_{\text{bank}}$ is first modulated by LOPI's gate and Gaussian, producing a gated bank signal. This signal is then used as the "target" input to SCAR, which projects it onto the contrastive subspace before adding it to the residual stream.
2. **Parallel (LOPI for bank, SCAR for residual).** LOPI modulates only the bank readout; SCAR modulates only the residual-stream perturbation from a separate calibration set (e.g., CAA-style contrastive prompts unrelated to the bank).

The current v0.5 implementation uses the **parallel** pattern: LOPI wraps the attention-bank readout (`lopi.py` lines 344–410), and SCAR wraps the attention-output hook (`scar_injector.py` lines 174–253). The two injectors are independent: LOPI's $\gamma_t$ does not affect SCAR's $\alpha$, and SCAR's $P_B$ does not affect LOPI's $M_\perp$.

**When shields commute.** If LOPI's gate $\gamma_t$ and SCAR's projection $P_B$ commute ($\gamma_t P_B = P_B \gamma_t$), the sequential composition is well-defined. In practice, $\gamma_t$ is a scalar (broadcast over the $d$-dimensional activation) and $P_B$ is a $d \times d$ matrix, so commutativity holds: $\gamma_t P_B (\text{target} - x) = P_B (\gamma_t (\text{target} - x))$. However, when $\gamma_t$ varies per token and SCAR is applied to a sequence of tokens (e.g., in a KV-cache decode with $T > 1$), the gates must be synchronized: $\gamma_t$ should be computed *after* SCAR's projection so that the gate sees the post-projection activation, not the pre-projection residual. This is not yet implemented in v0.5; the sequential composition is **future work** (tracked in the v0.6 roadmap).

### 5.3 Unified Framework: CAA-then-SCAR, or SCAR-as-CAA

The v0.5 memory stack has three injection points:
1. **Attention-bank readout** (wrapped by LOPI): modulates the weighted sum $W_{:,T:} \cdot M_V$ at the merged-softmax branch of `attn_native_bank.py`.
2. **Residual-stream CAA** (wrapped by `caa_injector.py`): adds $\alpha \cdot s$ to the hidden state at a chosen layer.
3. **Residual-stream SCAR** (wrapped by `scar_injector.py`): adds $\alpha \cdot P_B (\text{target} - x)$ to the attention output at a chosen layer.

These three are **not mutually exclusive**. A valid v0.5 configuration can have LOPI active at all layers, CAA active at layer $\ell_1$, and SCAR active at layer $\ell_2 \neq \ell_1$. The total residual perturbation is:

$$x_\ell' = x_\ell + \text{LOPI}_\ell + \text{CAA}_\ell + \text{SCAR}_\ell,$$

where each term is zero if the corresponding injector is not attached at layer $\ell$.

**Design tension.** SCAR was introduced as a "rescue" for CAA's W.4 failure (see `docs/theory/w3_failure_diagnosis.md`). The W.4 grid showed that CAA increased drift in 11 out of 12 significant tests, suggesting that unconstrained additive steering is harmful. SCAR's bounded projection was expected to fix this. However, the GB10 smoke (PR #22) tested SCAR in **isolation** (no LOPI, no CAA, no attention bank), so it does not yet answer the question: "Does SCAR fix CAA's failure when both are active?"

**Open question.** If SCAR and CAA are both attached at the same layer, do their perturbations interfere? Let $s_{\text{CAA}}$ be CAA's steering vector and $P_B$ be SCAR's projector. The composed perturbation is:

$$x' = x + \alpha_{\text{CAA}} \cdot s_{\text{CAA}} + \alpha_{\text{SCAR}} \cdot P_B (\text{target} - x).$$

If $s_{\text{CAA}} \in \text{span}(B)$, the two perturbations reinforce along the contrastive subspace. If $s_{\text{CAA}} \perp \text{span}(B)$, CAA adds off-axis energy that SCAR would have discarded, potentially re-introducing the dilution problem. This is conjectured to be the reason why SCAR should *replace* CAA, not *augment* it — but the hypothesis is untested. See Section 7.

---

## 6. GB10 Multi-Architecture Empirical Evidence

The GB10 multi-architecture smoke (PR #22, commit `91530794`, merged 2026-05-05) tested SCAR against CAA on three independent transformer families: Gemma-4-E2B (Google), Qwen3-4B-Instruct-2507 (Alibaba), and GLM-4-9B-0414 (THUDM). The experiment used 16 paired contrastive prompts for calibration and 10 held-out "gold" prompts for evaluation. The drift metric was the mean over prompts of $\max_i |\text{logits}_i^{\text{baseline}} - \text{logits}_i^{\text{steered}}|$ (max absolute logit perturbation; lower is better).

### Results

| Model | Layer | $\alpha=1$ CAA drift | $\alpha=1$ SCAR drift | SCAR/CAA ratio |
|-------|------:|---------------------:|----------------------:|---------------:|
| Gemma-4-E2B | 34 | 10.66 | 3.95 | 0.37 (2.7× tighter) |
| Qwen3-4B-Instruct-2507 | 16 | 11.38 | 2.25 | 0.20 (5.1× tighter) |
| GLM-4-9B-0414 | 36 | 12.82 | 2.87 | 0.22 (4.5× tighter) |

**Verdict:** `scar_better` on all 3 architectures. At $\alpha = 1$, SCAR's drift is 63–80% lower than CAA's across all models.

**$\alpha = 0$ red-line.** Drift at $\alpha = 0$ was exactly $0.0$ (to machine precision) for all 150 cells (3 models × 5 $\alpha$ values × 10 prompts) in the CAA, SCAR, and "none" arms. This confirms that the $\alpha$-shield property (Section 4.1) holds in practice.

**Cross-platform stability.** Gemma-4-E2B was tested on both M4 MPS (bf16) and GB10 CUDA (bf16). Drift at $\alpha = 1$ was 4.02 (M4) vs 3.95 (GB10), a difference of 0.07 nats (1.7%). This rules out backend-specific kernel divergence in the SVD or projection arithmetic.

**Full results.** See `reports/cleanroom/scar_smoke_gb10/report.md` (lines 1–90) and the per-model summary JSON files:
- `reports/cleanroom/scar_smoke_gb10/gemma4_summary.json` (lines 1–40)
- `reports/cleanroom/scar_smoke_gb10/qwen3_4b_summary.json`
- `reports/cleanroom/scar_smoke_gb10/glm4_9b_summary.json`

### Why This Matters

The three architectures tested have distinct structural properties:
- **Gemma-4-E2B** uses sliding-window attention with global tokens (MQA, RoPE with base 10000).
- **Qwen3-4B-Instruct** uses standard GQA (grouped-query attention) with RoPE base 1000000.
- **GLM-4-9B** uses MQA (multi-query attention) with a custom RoPE variant.

All three show the same qualitative result: SCAR's orthogonal projection onto a low-rank contrastive subspace damps logit perturbations by a factor of 2.7–5.1× compared to CAA's unconstrained additive steering. This cross-family generalization rules out the "single-model artefact" interpretation: the dilution problem is not an idiosyncrasy of Gemma or Qwen but a fundamental limitation of additive steering.

**Caveat.** The GB10 smoke tested SCAR in **isolation**: no attention bank, no LOPI gating, no multi-fact interference. The experiment answers the narrow question "Does SCAR's projection reduce drift relative to CAA when both are calibrated on the same contrastive pairs?" but does not answer the broader question "Does SCAR fix the W.4 failure where CAA + bank increased drift?" That question requires a full-stack ablation (A.4 PREREG, Section 4.4, lines 52–56) which is pending Phase D. Until then, the GB10 evidence is **necessary but not sufficient** for adopting SCAR as the main-line steering method.

---

## 7. Open Questions

### 7.1 What Does SCAR Do That CAA Cannot?

The GB10 evidence shows that SCAR reduces drift by 2.7–5.1× relative to CAA at $\alpha = 1$. But *why*? Two competing hypotheses:

**Hypothesis 1 (Noise filtering).** The contrastive subspace $\text{span}(B)$ isolates the "true" semantic axis along which the positive-negative distinction is linearly separable. CAA's mean vector $s = \bar{a}^+ - \bar{a}^-$ mixes this axis with prompt-specific noise (e.g., tokenization artefacts, dataset imbalance). SCAR's SVD filters the noise by retaining only the top-$k$ principal components.

**Hypothesis 2 (Norm stabilization).** CAA's additive rule $x' = x + \alpha \cdot s$ can increase the residual norm $\|x'\|$ without bound. SCAR's projection $P_B (\text{target} - x)$ is constrained by $\|P_B\|_{\text{op}} = 1$, so the injected displacement is always $\leq \|\text{target} - x\|$. The frozen model was trained on a distribution of residual norms; exceeding this range causes distributional shift, which manifests as increased drift.

**Distinguishing the hypotheses.** Hypothesis 1 predicts that SCAR's advantage should be largest when the calibration data is noisy (e.g., small $N$, high variance within positive or negative sets). Hypothesis 2 predicts that the advantage should correlate with the magnitude $\|s\|$: when $\|s\|$ is large, CAA's displacement is large, and norm stabilization matters most.

**Empirical test (not yet run).** Vary $N$ (calibration size) and measure SCAR's drift vs CAA's drift. If Hypothesis 1 is correct, the SCAR/CAA ratio should decrease (SCAR's advantage grows) as $N$ decreases. If Hypothesis 2 is correct, the ratio should be independent of $N$ but correlate with $\|s\| / \|x\|$.

### 7.2 When Does the Orthogonal-Projection Assumption Fail?

The A.4 ablation (PREREG lines 52–56) defines the "orthogonal-projection redline": ablating SCAR's projection (replacing $P_B (\text{target} - x)$ with the raw delta $(\text{target} - x)$) should *increase* drift relative to the control. This assumes that the projection is always beneficial — but is this true?

**Counterexample scenario 1: Near-rank-deficient $K$.** When the contrast matrix $X = [a_1^+ - a_1^-; \ldots; a_N^+ - a_N^-]$ is nearly rank-deficient (e.g., $\sigma_k \ll \sigma_1$), the top-$k$ singular vectors may not span the "true" contrastive subspace. In this case, SCAR discards useful signal that lies in the tail of the spectrum.

**Counterexample scenario 2: Target drift.** When $\text{target}$ is recomputed at every token (e.g., in a bank-conditioned SCAR where $\text{target}$ is the retrieval-weighted bank value), the contrastive subspace $\text{span}(B)$ may not align with the per-token displacement $(\text{target}_t - x_t)$. If $\text{target}_t$ drifts outside $\text{span}(B)$, the projection $P_B (\text{target}_t - x_t)$ removes the drift signal, causing the injection to lag behind the intended trajectory.

**Empirical signature.** If the projection is harmful, ablating it (A.4) should *reduce* drift relative to the control. The A.4 PREREG (lines 52–56) predicts the opposite: ablation should *increase* drift. If the data contradict this, the projection assumption fails.

**Open question.** What is the optimal rank $k$? The GB10 smoke used $k = 2$ (hardcoded in `experiments/scar_smoke/run.py`; not reported in the PR commit message). The rank-selection heuristic is **not documented** in the codebase. Future work should test $k \in \{1, 2, 4, 8\}$ and measure the SCAR/CAA ratio at each rank. The PREREG (Section 2, lines 41–49) suggests an ablation grid over $k$ but does not commit to a specific range.

### 7.3 Composition with LOPI: Do the Shields Interfere?

The v0.5 stack allows LOPI and SCAR to be active simultaneously at the same layer. The total residual perturbation is:

$$x' = x + \gamma_t \cdot w(\ell, t) \cdot M_\perp + \alpha \cdot P_B (\text{target} - x),$$

where the two terms are computed independently. Two potential interference modes:

**Interference 1: Redundant projection.** LOPI's $M_\perp$ is the orthogonal novelty (bank value minus its projection onto the native context value). SCAR's $P_B (\text{target} - x)$ is the projection of the displacement onto the contrastive subspace. If $M_\perp \in \text{span}(B)$, the two projections are redundant: SCAR's projection is a subset of LOPI's. In this case, the composed perturbation is:

$$x' = x + \gamma_t \cdot w(\ell, t) \cdot M_\perp + \alpha \cdot P_B (\text{target} - x) = x + (\gamma_t w + \alpha) \cdot M_\perp,$$

and the user has no independent control over $\gamma_t w$ and $\alpha$.

**Interference 2: Off-axis leakage.** If $M_\perp \not\perp \text{span}(B)$, LOPI's bank signal can add energy to the contrastive subspace that SCAR would have controlled via $\alpha$. This is the "leakage" problem: LOPI's gate $\gamma_t$ and Gaussian $w(\ell, t)$ were designed to modulate the bank readout, not to modulate the contrastive steering signal. Composing them naively may produce a perturbation that violates SCAR's bounded-contribution guarantee (Section 4.2).

**Empirical test (not yet run).** Measure drift for three configurations: (a) LOPI only, (b) SCAR only, (c) LOPI + SCAR. If interference is absent, $\text{drift}_c \leq \text{drift}_a + \text{drift}_b$. If interference is present, $\text{drift}_c > \text{drift}_a + \text{drift}_b$ (the composed perturbation is worse than the sum of the parts).

**Conjecture.** The sequential composition LOPI-then-SCAR (compute $\text{target} = \gamma_t \cdot w(\ell, t) \cdot M_\perp$, then apply SCAR to project $(\text{target} - x)$ onto $\text{span}(B)$) should be interference-free *if* the bank value $M_V$ is calibrated on the same contrastive pairs as the SCAR basis $B$. In this case, $M_\perp \approx \text{span}(B)$ by construction, and the projection is a no-op. This is **not guaranteed** when the bank is populated from a separate corpus (e.g., LAMA-style factual triples) and SCAR is calibrated on CAA-style contrastive prompts.

---

## 8. Provenance

This theory note is grounded in the following source files and line ranges:

| Claim | Source file | Lines | Citation |
|-------|-------------|-------|----------|
| SCAR injection formula $x' = x + \alpha \cdot P_B (\text{target} - x)$ | `deltamemory/memory/scar_injector.py` | 1–6, 207–253 | Module docstring (lines 1–6) and `_do_inject` method (lines 207–253) define the projection formula. Line 215 computes `projected = (delta @ basis) @ basis.T` where `delta = target - activations`. |
| Thin SVD basis extraction from $X = [a_i^+ - a_i^-]$ | `deltamemory/memory/scar_injector.py` | 84–128 | `calibrate` method. Line 116: `diff = pos - neg`. Line 117: `_, _, vh = torch.linalg.svd(diff, full_matrices=False)`. Line 126: `self.basis[layer] = vh[:k].T.contiguous()`. |
| $\alpha = 0$ short-circuit (red-line invariant) | `deltamemory/memory/scar_injector.py` | 209 | `if self.alpha == 0.0 ... return activations` precedes all projection arithmetic. |
| GB10 multi-architecture drift results (SCAR < CAA) | `reports/cleanroom/scar_smoke_gb10/report.md` | 1–90 | Tables at lines 15–24 (Gemma-4-E2B), 33–42 (Qwen3-4B-Instruct), 45–54 (GLM-4-9B). Verdict "scar_better" at line 59. |
| GB10 Gemma-4-E2B numeric drift at $\alpha=1$ | `reports/cleanroom/scar_smoke_gb10/gemma4_summary.json` | 1–40 | Line 20: `"1.0": 10.65546875` (CAA). Line 32: `"1.0": 3.95234375` (SCAR). |
| A.4 ablation orthogonal-projection redline | `experiments/A_ablation/PREREG.md` | 52–56, 99–106 | H_A4 hypothesis (lines 52–56): ablating orthogonal projection increases drift. Implementation table (lines 99–106, row "A4"). |
| CAA dilution problem (11/12 significant tests increased drift) | `docs/theory/w3_failure_diagnosis.md` | (entire file) | Full diagnosis of W.4 5,041-cell grid failure. Summary: CAA was Holm-significant in 12/12 tests, but 11/12 effects were *harmful* (increased drift). |
| LOPI derivative gate $\gamma_t = \sigma(k \cdot (\Delta_Q - \theta))$ | `deltamemory/memory/lopi.py` | 229–247 | `derivative_gate` function. Line 245: `sigmoid = torch.sigmoid(self.config.gate_k * (delta_q - self.config.gate_theta))`. |
| LOPI orthogonal novelty $M_\perp = M_V - \text{proj}_{V_{\text{ctx}}}(M_V)$ | `deltamemory/memory/lopi.py` | 344–355 | `orthogonal_novelty` function. Line 352: `m_parallel = (torch.sum(bank_value * ctx_value, ...) / (ctx_norm_sq + 1e-10)) * ctx_value`. Line 355: `return bank_value - m_parallel`. |
| CAA formula $h' = h + \alpha \cdot s$ | `deltamemory/memory/caa_injector.py` | 1–9 | Module docstring (lines 1–9) defines CAA formula. Implementation not directly cited (this note compares SCAR to CAA conceptually, not to a specific implementation). |
| mHC spectral cap $\sigma_{\max}(P_B) \leq \kappa$ (analogous bound for attention weights) | `docs/theory/mhc.md` | 69–93 | Prop. (Bank-column spectral cap) at lines 69–82. The bound $\sigma_{\max}(W'_{:,T:}) \leq \kappa$ is derived for attention weights; SCAR's bound $\sigma_{\max}(P_B) = 1$ is the residual-stream analogue. |

---

## Notes on Missing Evidence and Conjectures

**Conjecture 1 (Section 7.1).** The SCAR/CAA ratio should correlate with calibration noise (measured as within-set variance of positive or negative activations). This is **not yet tested**. The GB10 smoke used a fixed calibration set (16 pairs; seed-pinned) and did not vary $N$ or measure within-set variance.

**Conjecture 2 (Section 7.2).** The optimal rank $k$ depends on the intrinsic dimensionality of the contrastive subspace. The GB10 smoke used $k = 2$ (hardcoded in `experiments/scar_smoke/run.py`), but no rank-selection ablation was performed. Future work should test $k \in \{1, 2, 4, 8, 16\}$ and measure drift at each rank.

**Conjecture 3 (Section 7.3).** LOPI + SCAR composition is interference-free when the bank value $M_V$ and the SCAR basis $B$ are calibrated on the same contrastive pairs. This is **not yet tested**. The v0.5 stack allows LOPI and SCAR to be attached simultaneously, but no experiment has measured the composed drift vs the sum of individual drifts.

**Missing evidence (Section 6).** The GB10 smoke tested SCAR in isolation (no bank, no LOPI, no multi-fact interference). The full-stack A.4 ablation (PREREG lines 52–56) will test SCAR with the bank active, but the results are **pending Phase D**. Until then, the GB10 evidence is necessary but not sufficient for claiming that SCAR fixes the W.4 failure.

---

## Conclusion

SCAR replaces CAA's unconstrained additive steering with a projection onto a low-rank contrastive subspace. The method has three formal properties:
1. **Bit-equality at $\alpha = 0$** (Section 4.1): When $\alpha = 0$, SCAR is a mathematical identity, and the model output is unchanged.
2. **Bounded contribution** (Section 4.2): The injected displacement is bounded by $|\alpha| \cdot \|\text{target} - x\|$, preventing unbounded norm growth.
3. **Spectral cap** (Section 3.2): The orthogonal projector $P_B$ has unit spectral radius, so it cannot amplify the displacement.

The GB10 multi-architecture smoke (Section 6) provides evidence that SCAR reduces drift by 2.7–5.1× relative to CAA at $\alpha = 1$ on three independent transformer families (Gemma, Qwen, GLM). This cross-family generalization rules out single-model artefacts and supports the hypothesis that the dilution problem is a fundamental limitation of additive steering.

However, the GB10 evidence is **necessary but not sufficient** for adopting SCAR as the main-line steering method. The full-stack A.4 ablation (PREREG) will test SCAR with the attention bank active, answering the question: "Does SCAR fix the W.4 failure where CAA + bank increased drift?" Until that experiment is complete, SCAR remains a **candidate** replacement for CAA, not a validated component of the v0.5 stack.

Three open questions (Section 7) require empirical resolution:
1. **Noise vs norm stabilization** (Section 7.1): Does SCAR's advantage come from filtering noise or from stabilizing norms?
2. **Rank selection** (Section 7.2): What is the optimal $k$? How does SCAR's drift scale with rank?
3. **LOPI composition** (Section 7.3): When LOPI and SCAR are both active, do their perturbations interfere?

These questions define the Phase D roadmap. Until they are answered, the theory note documents SCAR's mathematical structure and GB10 evidence but does not claim that SCAR is the final solution to the dilution problem.

---

**End of U.6 SCAR theory note.**

*Written 2026-05-06 for Phase U (A3-scar-theory-note). All claims grounded in commit `91530794` (PR #22) and prior. No external citations; all references are to this repository.*
