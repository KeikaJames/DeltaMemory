# DeltaMemory Design (v2 / v3)

DeltaMemory is a **zero-parameter, attention-native** external memory for frozen
decoder LLMs. It does not train the base model. It does not rewrite weights.
It does not retrieve text into the prompt. The mechanism is one extra concat
inside every attention layer.

The Python package is `deltamemory` (formerly `rcvhc`).

## One-line core (v2)

For every non-shared attention layer ℓ in a frozen transformer:

```
Attn_ℓ(Q, [K ; M_K^(ℓ)], [V ; α · M_V^(ℓ)])
```

where `(M_K^(ℓ), M_V^(ℓ))` are tensors the model itself produced during a
**single forward pass over a write prompt**, captured at a chosen position
(post-norm, pre-RoPE for Gemma-style models). At read time, the model's
own softmax decides whether to attend to the bank. When `α = 0` or when
the bank is empty, the output is **bit-for-bit equal** to the unpatched
model.

There is no encoder. No KeyProjector. No broadcast residual. No fine-tune.

## v3 = v2 + Stage 14 (refines K-space matching)

v2 fails on `Q: who is X? A:` style queries because the write-time K
(captured at the period of `"X is Y."`) and the read-time Q (at the `A:`
token) live in different regions of K-space. v3 fixes this with three
strictly additive layers, each preserving the α=0 bit-equality property:

### 14A — InfoNCE K-projector

A tiny `nn.Linear(d, d)` per layer applied **only to bank K** (never to the
sequence Q or K). Trained by InfoNCE on (canonical write prompt, paraphrase
query) positive pairs:

```
loss = -log( exp(sim(Q_q, proj(K_w)) / τ) /
             Σ_neg exp(sim(Q_q, proj(K_neg)) / τ) )
```

Negatives = same batch other facts + decoy relations. α=0 still bit-equal
because at α=0 the bank concatenation is a zero-width slice.

### 14B — Address-conditional capture

Replace "capture K at period token of write prompt" with "capture K at the
address span of the write prompt". For a write prompt
`"The mayor of Paris is Anne Hidalgo."`, the address span is `"The mayor of
Paris"` (regex `^(.+?) is .+?\.`); we capture K at the last token of that
span. Read-time queries like `"Q: Who is the mayor of Paris? A:"` hit
softer K-space neighborhoods of the same concept.

### 14C — Multi-position write

`policy="multi"` writes `(K_addr, V_target)` and `(K_period, V_target)`
both. Bank size doubles; retrieval radius widens.

### 14E — ROME-style writer rebuild

13C's pure-nullify approach failed at held-out 0.184. v3 rebuilds the
writer in closed form on the train set:

```
W_v_delta = (K_addr^T K_addr + λI)^-1 K_addr^T V_target
M_V := W_v_delta · K_addr   # at injection time
```

This is the same identity ROME uses for weight-edits, but applied to a
non-destructive bank slot rather than the model's MLP rows.

## Position-agnostic invariant (Gemma-4)

Gemma-4 uses RoPE per layer and KV-shared layers. We capture K
**post-norm but pre-RoPE** so the bank carries no positional signal.
At read time we use a `q_pre` (pre-RoPE) copy for the bank scoring branch,
and `q_post` (RoPE-applied) for the standard sequence branch. KV-shared
layers route bank lookups through `kv_shared_layer_index` so all 35 / 35
attention layers see the bank (vs. only the 15 non-shared ones).

## ArchAdapter (multi-model)

`AttnNativePatcher` dispatches to a per-architecture `ArchAdapter` that
encapsulates the family-specific quirks:

| family | q_norm/k_norm | v_norm | KV-shared | RoPE base |
|---|---|---|---|---|
| Gemma-4 | yes | yes | yes (per-layer) | 1e4 |
| Qwen3 | yes | no | no | 1e6 |
| Llama / DeepSeek / Mistral | no | no | no | 1e4 |
| GLM-4 | no | no | no | 1e4 |

Each adapter must satisfy the same four unit gates:

1. Empty-bank forward bit-equal (max-abs-diff = 0.0).
2. α=0 with non-empty bank bit-equal.
3. α>0 single-fact write/read produces a measurable target-rank lift.
4. State-dict round-trip preserves all (K, V, head_dims, metadata).

## Storage and capacity

Per fact, the bank stores per-layer `(K, V)` tensors at
`(num_kv_heads, head_dim_layer)` resolution. For Gemma-4-E2B at
N=100 facts × 35 layers × kv_heads=1 × head_dim ∈ {256,512} ×
bf16 = ~10 MB. The bank lives on the model device; no CPU swap, no disk.

Capacity is not the bottleneck — read-time softmax dilution is. v2 zero-shot
useful-capacity is ~10 facts. v3 (with 14A InfoNCE projector) targets
useful-capacity ≥ 100 facts on Gemma-4-E2B at recall@1 ≥ 0.55.

## What this is not

- **Not RAG.** Retrieved text is never put into the prompt. The query
  prompt at read time contains only the address.
- **Not MEMIT / ROME (weight-edit).** The base model weights are frozen.
  v3's "ROME-style" name refers only to the closed-form least-squares
  solve used to compute `W_v_delta`; the result is stored in the bank,
  not written to MLP rows.
- **Not fine-tune.** v3 trains exactly one tiny `nn.Linear(d, d)` per
  layer (the InfoNCE K-projector). The base model receives zero gradient.
- **Not prompt insertion.** No fact text is added to any prompt.

## Trace requirements

Every benchmark run records:

- **Per-fact**: target-token rank, target-token logit, bank slot index,
  capture position, retrieval softmax weight.
- **Per-layer**: bank size N, head_dims, K/V dtypes, max-abs-diff to
  no-bank baseline at α=0.
- **Per-suite**: recall@1, recall@5, paraphrase robustness, decoy curve,
  long-context drift, locality probe drift, Wilcoxon W and p, Holm-corrected
  p-family, bootstrap 95% CI, Cohen's d.
- **Reproducibility**: model + tokenizer SHA, transformers version,
  device + dtype, seed, holdout split SHA, preregistration commit SHA.

## Scientific-claim boundary

A v3 result is reported as **PASS** only if all of:

1. Hypothesis was committed to `docs/preregistration.md` *before* the test
   set was touched.
2. Wilcoxon p < α / k_holm against every required baseline.
3. The result holds across ≥ 3 seeds.
4. Locality probe drift ≤ 0.05 (no collateral damage).

Anything else is reported honestly as FAIL or as a diagnostic, never
silently dropped or rerun until it passes.
