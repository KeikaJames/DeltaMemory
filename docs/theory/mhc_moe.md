# mHC × MoE: Why the Current Column-Cap is Wrong on Mixture-of-Experts Models

**Phase U.2** — DeltaMemory theory documentation.

---

## 1. The Problem in One Sentence

DeltaMemory's mHC shield applies a **global** bank-column cap across all queries simultaneously. In a Mixture-of-Experts (MoE) model each token's attention output is a **router-gated sum over expert sub-outputs**. A global cap conflates attention weights from different experts, producing over-suppression on some experts and under-suppression on others. **The current implementation has never been tested on any MoE model.**

---

## 2. MoE Attention Routing — Three Architectures

### 2.1 Standard MoE Attention (Mixtral 8×7B; Jiang et al. 2024)

Mixtral uses a dense attention layer per transformer block — the MoE routing applies to the FFN, not the attention. However, the *output projection* $W_o^{(e)}$ is per-expert in some variants. More importantly, in architectures where V projections are expert-specific (e.g. some implementations of Switch Transformer, Fedus et al. 2021), the bank-injection problem is immediate.

For a standard MoE-attn block with expert-specific V projections, the attention output for token $i$ under expert $e$ is:

$$\text{out}^{(e)}_i = \sum_{j=1}^{T} W_{ij} V^{(e)}_j + \sum_{n=1}^{N} W_{i,T+n} \cdot \alpha M_V^{(e)}_n,$$

and the final token output is the router-gated mixture:

$$\text{out}_i = \sum_e g_e(i) \cdot \text{out}^{(e)}_i,$$

where $g_e(i) \geq 0$ and $\sum_e g_e(i) = 1$ are the router soft gates for token $i$.

### 2.2 KV-Shared MoE (Qwen3-MoE; Qwen 2024)

Qwen3-MoE uses shared K/V projections across experts (only Q and the FFN experts are per-expert). The attention weights $W_{ij}$ are therefore the same for all experts. However, the bank M_V is written from the **shared V projection** of a single model forward — it is not factored per-expert. The effective per-expert bank contribution is:

$$\text{out}^{(e)}_{\text{bank}} = g_e(i) \cdot \sum_{n} W_{i,T+n} \cdot \alpha M_V_n.$$

A global column-sum cap treats all tokens equally regardless of $g_e(i)$, so the cap is applied to the wrong distribution.

### 2.3 MLA (Multi-head Latent Attention; DeepSeek-V3; DeepSeek 2024)

DeepSeek-V3 uses MLA: a low-rank compressed KV latent $c_t^{KV} \in \mathbb{R}^{d_c}$ is the stored cache, and K/V are up-projected from it. The expert MoE routing in DeepSeek-V3 applies to the FFN layers; attention is dense. Nevertheless, the attention output is routed to different experts by the following FFN stage, so the *effective* bank injection energy seen by each downstream expert depends on which tokens were routed where.

For MLA the bank score is computed against the shared compressed latent:

$$\text{scores}^{\text{bank}}_{i,n} = q_i^{\text{pre}} \cdot M_K^{(n)\top} \cdot s,$$

where $M_K^{(n)}$ is stored pre-RoPE at write time (same as the dense case). No per-expert structure changes the score computation, but the downstream routing implies the cap should still be expert-aware to maintain per-expert energy bounds.

---

## 3. Why the Global Cap is Wrong

### 3.1 Formal statement of the bug

Let $W \in \mathbb{R}^{q \times (T+N)}$ be the post-softmax weight matrix for a batch of $q$ queries from *all* experts combined. The current shield computes:

$$\text{col\_sum}_n = \sum_{i=1}^{q} W_{i,T+n}$$

and caps uniformly. But query $i$ was routed to expert $e_i$ with gate $g_{e_i}(i)$. The effective column sum for expert $e$ is:

$$\text{col\_sum}^{(e)}_n = \sum_{i : e_i = e} g_e(i) \cdot W_{i,T+n}.$$

If tokens routed to expert $e_1$ happen to all attend strongly to bank slot $n$, the global sum will trigger a cap that also rescales the rows of expert $e_2$'s tokens, even if $e_2$'s per-expert column sum was well within $\kappa$. Conversely, if $e_1$ tokens are diluted across many bank slots, their column sum may be low globally, evading the cap even though within-expert routing is concentrated.

### 3.2 Numerical illustration

Suppose $q = 4$ queries, $N = 1$ bank slot, two experts with $g_1 \in \{0.9, 0.9, 0.1, 0.1\}$ and $g_2 = 1 - g_1$. Let $W_{i,T+1} = 0.4$ for queries 1–2 (expert 1 tokens) and $W_{i,T+1} = 0.05$ for queries 3–4.

- **Global col-sum**: $0.4 + 0.4 + 0.05 + 0.05 = 0.9$. With $\kappa = 1.0$, the global cap does nothing.
- **Expert-1 effective col-sum**: $0.9 \times 0.4 + 0.9 \times 0.4 = 0.72$. Still below $\kappa = 1$.
- **But**: if $q = 10$ and all belong to expert 1 with $W_{i,T+1} = 0.15$, global col-sum = 1.5 → cap fires and rescales all queries by $1.0/1.5$, even though this batch has only one expert and the per-expert bound is exactly right. No harm. But if half are expert-1 ($W=0.25$) and half expert-2 ($W=0.05$), global sum = 1.5, cap applies $0.667×$ to all — expert-2 rows are needlessly shrunk by 33%.

---

## 4. The Correct Fix: Per-Expert Column Cap

### 4.1 Formula

For each expert $e$ define the **expert-masked bank weight**:

$$W^{(e)}_{i,T+n} = g_e(i) \cdot W_{i,T+n}.$$

Apply the column cap independently per expert:

$$\text{col\_sum}^{(e)}_n = \sum_{i=1}^{q} W^{(e)}_{i,T+n},$$

$$\text{scale}^{(e)}_n = \min\!\left(1,\ \frac{\kappa}{\text{col\_sum}^{(e)}_n}\right),$$

$$\tilde{W}^{(e)}_{i,T+n} = W^{(e)}_{i,T+n} \cdot \text{scale}^{(e)}_n.$$

The final shielded weight for query $i$ aggregates back:

$$\tilde{W}_{i,T+n} = \sum_e \tilde{W}^{(e)}_{i,T+n} / g_e(i) \quad (\text{re-normalize per query}).$$

This guarantees $\sum_i \tilde{W}^{(e)}_{i,T+n} \leq \kappa$ for every expert $e$ and every bank slot $n$, while the global column sum may exceed $\kappa$ if many experts each contribute up to $\kappa$.

### 4.2 Architecture-specific formulas

#### Standard MoE-attn (expert-specific V)

$$\text{out}_{\text{bank}} = \sum_e g_e \odot \left(\tilde{W}^{(e)}_{:,T:} \cdot \alpha M_V^{(e)}\right),$$

where $\odot$ denotes router-gated token masking. Each expert's bank cap is independent.

#### KV-Shared MoE (shared V, Qwen3-MoE style)

Since $M_V$ is shared, the per-expert effective injection is $g_e(i) \cdot \tilde{W}^{(e)}_{i,T+n} \cdot \alpha M_V_n$. The router gate is the natural "per-expert query mask" needed to separate col-sums.

#### MLA (DeepSeek-V3 style)

MLA attention is dense; the MoE routing is downstream in the FFN. The cap is still applied to the attention bank columns, but the correct bucketing is by the **next-layer expert assignment** (which is determined by the output of this attention layer). In practice, a conservative approximation is to apply a per-head cap (each head attends somewhat independently), with a global cap as a fallback. This is Phase W.5 territory.

---

## 5. Implementation Status

**Current status: not implemented.** The `shield_attention_weights` function in `mhc_shield.py` (lines 110–154) takes no argument for expert gates and applies a single global column cap. There is no per-expert dispatch in `attn_native_bank.py`. The entire MoE code path has **never been exercised in any experiment**.

Phase W.5 must:
1. Identify the router gate tensors for Mixtral / Qwen3-MoE / DeepSeek-V3 during the patched forward.
2. Pass per-token gate vectors alongside `weights` into `shield_attention_weights`.
3. Implement the per-expert bucketing formula above.
4. Run the W.1 style drift grid on at least one MoE model.

Until W.5 is complete, **mHC should be considered unvalidated on all MoE architectures**.

---

## 6. Summary

| Architecture | MoE V | Current cap | Correct cap |
|---|---|---|---|
| Dense (Gemma-4, Qwen2.5, Llama) | N/A | Global col-sum ✓ | Global col-sum ✓ |
| Mixtral 8×7B (FFN-MoE, dense attn) | Shared | Global (approximately OK) | Per-head |
| Qwen3-MoE (KV-shared, FFN-MoE) | Shared | Global ✗ | Per-expert (router gate) |
| DeepSeek-V3 MLA (FFN-MoE) | Shared via latent | Global ✗ | Per-expert (approx. per-head) |

---

*References:* Fedus et al., *Switch Transformer*, JMLR 2022; Jiang et al., *Mixtral of Experts*, arXiv:2401.04088, 2024; DeepSeek, *DeepSeek-V3*, arXiv:2412.19437, 2024; Qwen, *Qwen3 Technical Report*, 2024.
