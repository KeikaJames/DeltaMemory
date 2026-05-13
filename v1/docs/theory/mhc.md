# mHC: From Sinkhorn–Knopp to Bank-Column-Only Spectral Cap

**Phase U.1** — Mneme theory documentation.

---

## 1. The Birkhoff Polytope and Doubly-Stochastic Matrices

A matrix $W \in \mathbb{R}^{n \times n}$ is **doubly-stochastic** if

$$W_{ij} \geq 0, \quad \sum_j W_{ij} = 1 \; \forall i, \quad \sum_i W_{ij} = 1 \; \forall j.$$

The set of all $n \times n$ doubly-stochastic matrices is the **Birkhoff polytope** $\mathcal{B}_n$.

**Birkhoff–von Neumann Theorem** (Birkhoff 1946; von Neumann 1953): Every doubly-stochastic matrix is a convex combination of permutation matrices. Equivalently, the extreme points of $\mathcal{B}_n$ are exactly the $n!$ permutation matrices.

This theorem is the foundation of the spectral bound we exploit: permutation matrices are orthogonal, so their singular values are all 1; convexity and sub-multiplicativity then enforce $\sigma_{\max}(W) \leq 1$ for all $W \in \mathcal{B}_n$.

---

## 2. Sinkhorn–Knopp Convergence

**Sinkhorn (1967)** proved that any strictly positive matrix can be scaled to a doubly-stochastic matrix by alternating row-normalisation and column-normalisation. Concretely, initialise $W^{(0)} = A$ (positive entries) and iterate:

$$W^{(2t+1)} = D_r^{-1} W^{(2t)}, \quad D_r = \operatorname{diag}(W^{(2t)} \mathbf{1}),$$
$$W^{(2t+2)} = W^{(2t+1)} D_c^{-1}, \quad D_c = \operatorname{diag}(\mathbf{1}^\top W^{(2t+1)}).$$

Convergence to the unique doubly-stochastic scaling $D_1 A D_2$ is guaranteed for strictly positive matrices (Sinkhorn 1967; Bregman 1967).

**Cuturi (2013)** showed that adding entropic regularisation $\lambda H(W)$ transforms the Wasserstein transport problem into a series of Sinkhorn iterations on a kernel matrix $K_{ij} = e^{-C_{ij}/\lambda}$, enabling GPU-friendly $O(n^2)$ optimal-transport approximations. This motivated widespread use of Sinkhorn in modern deep learning.

Our `sinkhorn_knopp_projection` in `mhc_shield.py` implements vanilla SK for unit-testing the math (lines 64–107). **It is not used in production** — see Section 4.

---

## 3. Doubly-Stochastic Spectral Property

**Theorem.** For any doubly-stochastic $W \in \mathcal{B}_n$, $\sigma_{\max}(W) \leq 1$, with equality iff at least one row or column degenerates to a one-hot vector.

**Proof (3 lines).**
(i) For any vector $x$: $\|Wx\|_2^2 \leq \|W\|_\infty \|W\|_1 \|x\|_2^2$ by Hölder + sub-multiplicativity of induced norms.
(ii) Row-stochasticity gives $\|W\|_\infty = 1$; column-stochasticity gives $\|W\|_1 = 1$.
(iii) Therefore $\|W\|_2 \leq \sqrt{\|W\|_\infty \|W\|_1} = 1$. ∎

Equality holds (e.g. for any permutation matrix) precisely when $\|Wx\| = \|x\|$ for some $x$, which requires every row and column to be one-hot (no mixing).

**Numerical counterexample.** Take

$$W = \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.5 \end{pmatrix} \in \mathcal{B}_2.$$

Row sums = column sums = 1. Singular values are $\{1, 0\}$, so $\sigma_{\max} = 1 < \sqrt{2}$. Contrast with the identity $I$: singular values $\{1, 1\}$, $\sigma_{\max} = 1$ (equality, each row and column is one-hot).

---

## 4. Why We Did NOT Use Full Sinkhorn–Knopp

Three independent reasons preclude applying the full SK projection to the merged attention weight matrix $[W_{\mathrm{seq}}; W_{\mathrm{bank}}]$:

1. **Contaminates native attention.** SK's column-normalisation step rescales the *sequence columns* $W_{:,:T}$, which the frozen LLM was trained to interpret as row-normalised softmax outputs. Disturbing those weights was empirically catastrophic: applying full SK on the merged matrix produced +5 nats NLL drift on Gemma-4-E2B (observed during Phase R; documented in `mhc_shield.py` lines 11–16).

2. **$O((N+T) \cdot \text{iters})$ iteration cost.** Even with only 3 SK rounds, the inner loops touch every element of the $q \times (T+N)$ weight matrix at every attention layer of every forward step. For autoregressive decode with $T$ growing to thousands, this is prohibitive for a regulariser that is already serving a secondary role.

3. **Breaks $\alpha = 0$ bit-equality.** When `alpha = 0` the injection branch is skipped entirely (short-circuit at the call site), so the shield never runs and the model output is bit-for-bit identical to the unpatched baseline. Full SK on the merged matrix would destroy this invariant even when the bank contributes nothing, violating the red-line requirement documented in `mhc_shield.py` lines 33–35.

---

## 5. Column-Cap on Bank Columns Only

### Formal Proposition

**Prop. (Bank-column spectral cap).**  
Let $W \in \mathbb{R}^{q \times (T+N)}$ be the row-stochastic post-softmax attention weight matrix, with the first $T$ columns indexing native sequence K/V slots and the last $N$ columns indexing bank slots. After the shield operation (cap each bank column sum to $\kappa$):

$$\sum_{i=1}^{q} W'_{i, T+j} \leq \kappa \quad \forall j \in \{0, \ldots, N-1\}.$$

Then, by Gershgorin / matrix norm sub-multiplicativity:

$$\sigma_{\max}(W'_{:,T:}) \leq \sqrt{\kappa \cdot \max_j \textstyle\sum_i W'_{i,T+j}} \leq \sqrt{\kappa \cdot \kappa} = \kappa.$$

Native columns $W'_{:,:T}$ are returned **bit-for-bit unchanged**.

### Why This Bounds the Injection Energy

The bank readout is

$$\text{out}_{\text{bank}} = W'_{:,T:} \cdot (\alpha M_V),$$

so by the above bound and submultiplicativity of the Frobenius norm:

$$\|\text{out}_{\text{bank}}\|_F \leq \sigma_{\max}(W'_{:,T:}) \cdot \|\alpha M_V\|_F \leq \kappa \cdot \|\alpha M_V\|_F.$$

The $\alpha$ factor enters only through $M_V$'s magnitude; the cap is applied *after* softmax, removing $\alpha$-dependent score saturation. (See `mhc_shield.py` lines 38–48 for the in-code mathematical statement.)

---

## 6. Boundary Conditions and Known Holes

### 6.1 Row-stochastic property is broken

After capping, bank columns satisfy $\sum_i W'_{i,T+j} \leq \kappa < 1$ (when $\kappa < 1$), so the row sums of the full weight matrix satisfy

$$\sum_{k=1}^{T+N} W'_{ik} = \underbrace{\sum_{k=1}^{T} W_{ik}}_{<1 \text{ (since bank sums moved)}} + \sum_{j=1}^{N} W'_{i,T+j} < 1.$$

The "missing mass" $1 - \sum_k W'_{ik}$ is an implicit **do-nothing / sink token** — the query attends partly to nothing. This is a form of **signal attenuation**: the residual stream receives less total value energy per query than the softmax originally requested. The magnitude of this effect scales with $N$ and is worst when $N$ is large and $\kappa$ is small. Quantification is left to Phase W.1.4 (DH2 diagnostic).

### 6.2 Cap constrains routing, not energy

The column cap bounds $\sigma_{\max}(W_{\text{bank}})$ but says nothing about $\|M_V\|$. If the bank V vectors are large (e.g. Qwen2.5 without native `v_norm` can have $\|M_V\|$ 5–10× larger than Gemma-4), the injection energy $\|\text{out}_{\text{bank}}\|_F \leq \kappa \cdot \alpha \|M_V\|_F$ remains large. **mHC and V-scale (R-7) are two independent constraints that must both be active to fully bound injection energy.** mHC alone is necessary but not sufficient for cross-architecture $\alpha$-linearity.

---

## 7. Code Reference Table

| Mathematical object / theorem | File | Lines |
|---|---|---|
| Module docstring (math guarantee, design rationale) | `mhc_shield.py` | 1–57 |
| Doubly-stochastic definition (col-sum iteration) | `mhc_shield.py` | 100–107 |
| `sinkhorn_knopp_projection` (full SK, test only) | `mhc_shield.py` | 64–107 |
| Full SK is NOT called on merged matrix (rationale) | `mhc_shield.py` | 11–16, 72–75 |
| $\alpha=0$ bit-equality short-circuit reason | `mhc_shield.py` | 33–35 |
| `shield_attention_weights` (column-cap impl) | `mhc_shield.py` | 110–154 |
| `enabled=False` → identity (bit-equality) | `mhc_shield.py` | 140–141 |
| `bank_size=0` guard | `mhc_shield.py` | 142–143 |
| Bank column extraction + col_sum | `mhc_shield.py` | 149–150 |
| Cap + clamp + dtype restore | `mhc_shield.py` | 151–154 |
| Gershgorin / sub-multiplicativity math statement | `mhc_shield.py` | 38–48 |
| mHC shield call site in patched forward | `attn_native_bank.py` | 517–530 |
| `mhc_shield` flag on bank dataclass | `attn_native_bank.py` | 149–155 |

---

## 8. Summary

We started from the classical Birkhoff–von Neumann theorem ($\sigma_{\max} \leq 1$ for doubly-stochastic matrices), motivated by DeepSeek's mHC training-time constraint. We showed via Sinkhorn–Knopp that the full doubly-stochastic projection is the right mathematical object but the wrong inference-time operation. Our production solution is a minimal, parameter-free column-sum cap on the bank columns only, which (a) bounds the spectral norm of the bank injection operator at $\kappa$, (b) leaves native attention bit-equal, and (c) has $O(N)$ cost per layer. The residual weaknesses — implicit sink mass and unconstrained V energy — are the motivation for V-scale (R-7) and the W.1 experimental programme.

---

*References:* Birkhoff (1946); von Neumann (1953); Sinkhorn (1967); Cuturi, *Sinkhorn Distances*, NeurIPS 2013; Xie et al., *Manifold-Constrained Hyper-Connections*, arXiv:2512.24880, 2026.
