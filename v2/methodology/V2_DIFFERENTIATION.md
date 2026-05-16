# V2 Differentiation — Why ALB is Not X

> **Purpose**: Establish precise mathematical and operational distinctions between ALB (Attention-Side Latent Bank) and existing techniques (RAG, LoRA, ICL, CoT, MEMIT, Constitutional AI). Prevent category confusion.

---

## 1. ALB vs RAG (Retrieval-Augmented Generation)

### 1.1 Operational Difference

| Aspect | RAG | ALB |
|---|---|---|
| **What is retrieved** | Text strings (documents, passages, KB entries) | Pre-computed hidden states (h ∈ ℝ^d) from model's internal layers |
| **Where it enters** | Prompt (input token sequence) | AttentionBank → directly into K/V at target layer |
| **Re-tokenization** | YES — retrieved text is tokenized, embedded (E · tokens → h₀) | NO — bank entries bypass token embedding entirely |
| **Attention path** | Full L layers: retrieved text goes through embedding + all L decoder layers | Injected at layer l: bank h is projected (I+P)·h → W_K/W_V → concat to layer-l K/V only |
| **Prompt pollution** | Prompt length grows with retrieved context | Prompt stays minimal (just the question) |
| **Trainable** | External retriever (BM25, dense, learned) + frozen LM | Frozen LM + learnable K-projector P (rank-r) + gate heads |

### 1.2 Equation Difference

**RAG forward pass** (simplified, single-layer Transformer decoder):
```
Given query q, retriever R returns docs D = {d₁, …, d_k}
Prompt = concat(q, d₁, d₂, …, d_k)
Tokens = tokenize(Prompt) ∈ ℕ^T'   (T' > T_q due to context)
h₀ = E · Tokens                    (embedding layer)
for l = 1 to L:
    K_l = W_K^l · h_{l-1}          (all T' tokens)
    V_l = W_V^l · h_{l-1}
    Q_l = W_Q^l · h_{l-1}
    h_l = Attention(Q_l, K_l, V_l) + MLP(h_{l-1})
Output = LM_head(h_L)
```

**ALB forward pass** (LPL, K=2 rounds, bank injected at layer l_target):
```
Round 1:
  Tokens = tokenize(q) ∈ ℕ^T       (no retrieved docs appended)
  h₀ = E · Tokens
  for l = 1 to L:
      (standard forward, no bank interaction)
      # pause head may write to bank.slots[l] at paused positions

Round 2 (bank injection):
  h₀ = same as round 1 (or carry-over from round 1)
  for l = 1 to L:
      if bank.slots[l] non-empty:
          h_bank = bank.slots[l]                    ∈ ℝ^{N_b × d}
          # Project bank h with learnable K-projector P ∈ ℝ^{d × d} (or rank-r)
          h_bank_proj = h_bank + P(h_bank)          (residual projector)
          K_bank = W_K^l · h_bank_proj              (NO RoPE on bank K)
          V_bank = W_V^l · h_bank_proj
          # Compute self K/V from current h_{l-1} as usual
          K_self = W_K^l · h_{l-1}                  (with RoPE)
          V_self = W_V^l · h_{l-1}
          # Per-position bank gate g ∈ (0,1)
          g = σ(W_g^l · h_{l-1})                    ∈ ℝ^{T × 1}
          # Concat keys/values along sequence dimension
          K_concat = [K_self, K_bank]               ∈ ℝ^{(T + N_b) × d_k}
          V_concat = [V_self, V_bank]               ∈ ℝ^{(T + N_b) × d_v}
          # Softmax attention over combined K (bank slice has no causal mask)
          α = softmax(Q_l · K_concat^T / √d_k)      ∈ ℝ^{T × (T + N_b)}
          # Gate applied post-softmax on bank columns only
          α[:, T:] *= g                             (per-query modulation)
          h_l = α · V_concat + MLP(h_{l-1})
      else:
          (standard attention)
  Output = LM_head(h_L)
```

**Key distinction**: RAG augments the *input token sequence*, ALB augments the *attention key-value space at runtime* without changing prompt tokens.

---

## 2. ALB vs LoRA / PEFT / Fine-Tuning

### 2.1 Operational Difference

| Aspect | LoRA / Fine-Tuning | ALB |
|---|---|---|
| **Base model weights** | Modified (fine-tuning) or augmented with per-layer adapters (LoRA) | **Frozen** — W_Q, W_K, W_V, W_O, MLP, embeddings untouched |
| **Trainable params** | LoRA: rank-r adapters on all linear layers across L layers (~10M-100M params for LLaMA-7B, r=16-64) | K-projector P (rank-r, single shared or per-layer) + gate heads (~420K for Qwen3-4B, r=64, L=40) |
| **Storage per task** | LoRA: one adapter checkpoint per fine-tuned task (~10-100 MB) | ALB: one bank.pt (N_b × d vectors) + one P checkpoint (~1-5 MB) |
| **Inference** | Standard forward, but each linear layer has added adapter path | Multi-round forward with bank concat at target layers |
| **Memory source** | Implicitly encoded in adapter weights (not interpretable) | Explicitly stored as h-vectors in bank (can inspect/edit/inject) |

### 2.2 Numerical Comparison (Phase B2)

**LoRA (typical)** on Qwen3-4B (d=3584, L=40, r=64):
- Trainable: rank-64 on W_Q, W_K, W_V, W_O per layer + MLP down/up
  = 40 layers × (4 attn adapters + 2 MLP adapters) × 2 × (3584 × 64) = **≈110M params**

**ALB (Phase B2)** on Qwen3-4B:
- K-projector P (rank-64, shared): d × r + r × d = 3584 × 64 × 2 = **458,752 params**
- Bank_gate heads (per-layer, 40 heads): 40 × (3584 + 1) = **143,360 params**
- Pause heads (not trained in B2, but if trained): 40 × (3584 + 1) = **143,360 params**
- **Total (B2 config)**: 458,752 + 143,360 = **602,112 params** (vs 4B base)

**Ratio**: 0.6M / 4,000M = **0.015%** of base model size (vs LoRA's ~2.75%).

**Effect**: B2 achieved NLL 12.13 → 6.30 (Δ=−5.83) with 0.015% trainable params.

### 2.3 Conceptual Difference

- **LoRA/fine-tuning**: "Teach the model new behavior by adjusting internal computation."
- **ALB**: "Give the model external memory it can attend to; model's computation stays frozen."

ALB is closer to **external knowledge grounding** than model editing.

---

## 3. ALB vs In-Context Learning (ICL)

### 3.1 Operational Difference

| Aspect | ICL | ALB |
|---|---|---|
| **How examples provided** | Prefix prompt with few-shot examples (text) | Pre-fill bank with h-vectors from example encodings |
| **Prompt length** | T_prompt = T_query + k × T_example (grows linearly with k) | T_prompt = T_query (fixed, minimal) |
| **Example encoding** | Re-encoded every inference (k examples → k × T_ex tokens) | Encoded once, stored in bank.pt; reused across inferences |
| **Attention cost** | O(T²) where T includes all examples (quadratic in total length) | O(T_q × (T_q + N_b)) where T_q is query length only (bank N_b fixed) |
| **Example format** | Must be verbalized (text) | Can be latent (h-vectors from non-verbalizable states) |

### 3.2 Example

**ICL** for fact-recall:
```
Prompt:
"Paris is the capital of France.
 Tokyo is the capital of Japan.
 Berlin is the capital of Germany.
 What is the capital of Italy?"
```
- Tokens: ~50 (3 examples × ~12 tokens + query ~14 tokens)
- Each inference re-tokenizes and re-encodes all 3 examples.

**ALB** for fact-recall:
```
Bank preload (done once):
  - h₁ = encode("Paris is the capital of France.")[-1]  (last-token hidden at layer 9)
  - h₂ = encode("Tokyo is the capital of Japan.")[-1]
  - h₃ = encode("Berlin is the capital of Germany.")[-1]
  bank.slots[9] = stack([h₁, h₂, h₃])

Prompt (every inference):
  "What is the capital of Italy?"
```
- Tokens: ~8 (query only)
- Examples never re-encoded; attention reads bank.slots[9] directly.

**Cost difference**: ICL grows prompt by k examples → O((T_q + k·T_ex)²) attention. ALB keeps prompt fixed → O(T_q² + T_q·N_b).

### 3.3 Conceptual Difference

- **ICL**: "Examples are part of the conversational context; model learns task in-context."
- **ALB**: "Examples are external memory; model retrieves relevant memory via attention."

ICL and ALB can **co-exist**: a few high-level examples in prompt (ICL) + large knowledge base in bank (ALB).

---

## 4. ALB vs Chain-of-Thought (CoT) / Scratchpad

### 4.1 Operational Difference

| Aspect | CoT / Scratchpad | ALB |
|---|---|---|
| **What is generated** | Extra output tokens (reasoning steps as text) | NO extra output tokens — reasoning via extra *attention rounds* |
| **Token cost** | O(T_q + T_cot) tokens generated (T_cot can be 10-100× T_q for complex reasoning) | O(T_q) tokens generated (same as no-CoT baseline) |
| **Reasoning visibility** | Explicit (readable reasoning text in output) | Implicit (hidden-state dynamics in bank across rounds) |
| **Training** | Requires supervision on CoT chains (annotated reasoning) | No CoT supervision needed; only task loss (answer NLL) |
| **Inference latency** | Proportional to T_cot (sequential token generation) | Proportional to K_max (number of rounds, typically K=2-8) |

### 4.2 Example

**CoT** for "What is 37 × 24?":
```
Model output:
"Let's solve 37 × 24 step by step.
 First, 37 × 20 = 740.
 Then, 37 × 4 = 148.
 Finally, 740 + 148 = 888.
 The answer is 888."
```
- Tokens generated: ~50 (CoT reasoning) + 1 (final answer)
- User sees reasoning text.

**ALB (with K_max=4)** for same task:
```
Model output:
"888"
```
- Tokens generated: 1 (final answer only)
- Reasoning happens internally across K=4 rounds:
  - Round 1: encode question
  - Round 2: pause at layer 15, write intermediate h to bank
  - Round 3: read bank at layer 15, refine computation
  - Round 4: output final answer
- User does NOT see intermediate steps (hidden-state ponder).

**Cost**: CoT incurs O(T_cot) token generation; ALB incurs K × (forward pass overhead) but NO extra output tokens.

### 4.3 Equation Difference

**CoT** (auto-regressive token generation):
```
Prompt = "What is 37 × 24? Let's solve step by step."
for t in range(T_max):
    h_t = model(tokens[:t])
    token_t = sample(h_t[-1])
    tokens.append(token_t)
    if token_t == EOS:
        break
Output = tokens  (includes reasoning + answer)
```

**ALB multi-round** (no extra token generation):
```
Prompt = "What is 37 × 24?"
for k in range(K_max):
    h^k = forward_with_bank(prompt, round=k)
    p_halt = σ(W_halt · h^k[-1])
    if p_halt > 0.5:
        break
Output = sample(h^K[-1])  (single answer token, no reasoning text)
```

**Key distinction**: CoT generates reasoning *tokens*, ALB performs reasoning via *repeated attention with bank accumulation*.

---

## 5. ALB vs MEMIT / ROME (Model Editing)

### 5.1 Operational Difference

| Aspect | MEMIT / ROME | ALB |
|---|---|---|
| **What is edited** | Base model weights W (typically MLP layers) | NO weights edited — memory stored externally in bank |
| **Edit persistence** | Permanent (until model reloaded or next edit) | Session-based (bank cleared between inferences) OR persistent (preloaded LT bank) |
| **Edit mechanism** | Solve for Δ W such that W' · h_key = v_target (closed-form or gradient) | Learn projector P such that (I + P) · h_bank → queryable K/V |
| **Multiple edits** | MEMIT batches N edits, solves least-squares; >1000 edits degrade model | Bank holds N_b entries, no degradation (scales to 10K+) |
| **Read-time** | Edited knowledge is "baked in" — no explicit retrieval | Bank content explicitly retrieved via attention (gate-controlled) |

### 5.2 Technical Difference

**MEMIT** (simplified, MLP edit at layer l*):
```
Given: (subject, relation, target_new) to insert
1. Compute h* = encode(subject)[-1]  at layer l*
2. Compute v* = encode(target_new)[0]  (desired output embedding)
3. Solve for Δ W^l* such that:
       (W_MLP^l* + Δ W^l*) · h* ≈ v*
   subject to minimal change to W (ridge regression)
4. Update: W_MLP^l* ← W_MLP^l* + Δ W^l*
5. Model now "knows" (subject, relation, target_new) without re-training
```

**ALB** (Phase B2, static bank):
```
Given: (subject, relation, target_new) to insert
1. Compute h* = encode(subject + relation)[-1]  at layer l*
2. Compute b = MEMIT_residual(h*, target_new)  (from Exp35b, optional)
3. Store: bank.slots[l*].append(b)  (or preloaded from bank.pt)
4. Train projector P (rank-r) such that:
       W_K^l* · (I + P) · b → attends to correct query
       W_V^l* · (I + P) · b → retrieves correct target
5. Model weights W untouched; knowledge in bank, accessed via attention
```

**Key distinction**: MEMIT modifies W (permanent until reverted); ALB modifies bank (external, non-invasive).

### 5.3 Use-Case Difference

- **MEMIT**: "Correct a factual error in the model (e.g., 'Biden is president' → 'Trump is president')" → need persistence, few edits
- **ALB**: "Give the model access to a large, evolving knowledge base (e.g., user's email archive, codebase state)" → need scalability, session control

ALB cannot replace MEMIT for permanent single-fact corrections; MEMIT cannot replace ALB for large-scale external memory.

---

## 6. ALB vs Constitutional AI / System Prompts

### 6.1 Operational Difference

| Aspect | Constitutional AI / System Prompts | ALB |
|---|---|---|
| **What it controls** | Model's objective, alignment, safety guardrails, persona | Model's access to external memory (facts, context, working memory) |
| **Where it lives** | Prompt (system message, fixed prefix, or fine-tuning objective) | AttentionBank (hidden-state K/V augmentation) |
| **Orthogonality** | Completely orthogonal to ALB | Completely orthogonal to Constitutional AI |
| **Example** | "You are a helpful assistant. Never output harmful content." | bank.slots[9] = [h₁, h₂, …, h_N] (preloaded knowledge) |

### 6.2 Can They Co-Exist?

**YES**. Example combined system:
```
System Prompt (Constitutional AI):
  "You are a helpful coding assistant. Follow Python PEP8. Never execute unsafe shell commands."

ALB Bank (preloaded at layer 9):
  - Encoding of project README
  - Encoding of 500 most-used functions in codebase
  - Encoding of user's recent 50 edits (working memory)

User Query:
  "How do I refactor the auth module?"

Forward pass:
  - Round 1: encode query + system prompt (ICL-style)
  - Round 2: attention reads bank.slots[9] → retrieves relevant function encodings
  - Output: code suggestions (constrained by system prompt, informed by bank memory)
```

**Constitutional AI** = objective; **ALB** = mechanism. They serve different purposes and compose naturally.

---

## 7. Equation Appendix — Full Multi-Round LPL Forward

### 7.1 Notation

- L: number of decoder layers
- K_max: maximum rounds
- d: hidden size
- n_h: number of attention heads (query heads)
- n_kv: number of key/value heads (for GQA/MQA)
- d_h: head dimension (d / n_h)
- bank.slots[l]: stored h-vectors at layer l, shape [N_b, d]
- P^l: K-projector at layer l (shared or per-layer), rank-r or full-rank
- W_pause^l, W_gate^l, W_halt: learnable heads
- W_Q^l, W_K^l, W_V^l, W_O^l: frozen base Transformer weights

### 7.2 Forward Pass (Single-Layer, Round k)

**Input**: h_{in}^{k,l} ∈ ℝ^{B × T × d} (batch B, sequence length T)

**Step 1: Pause decision** (only if k ≥ 2 or pause head enabled)
```
p_pause^{k,l} = σ(W_pause^l · h_{in}^{k,l})  ∈ ℝ^{B × T}
if p_pause^{k,l}[b,t] > 0.5:
    bank.slots[l].append(h_{in}^{k,l}[b,t])    (write to bank)
    mask_skip[b,t] = True                       (skip this layer's attention for this position)
```

**Step 2: Self-attention** (skipped at masked positions)
```
h_norm = LayerNorm(h_{in}^{k,l})

Q = (W_Q^l · h_norm).view(B, T, n_h, d_h).transpose(1,2)    ∈ ℝ^{B × n_h × T × d_h}
K_self = (W_K^l · h_norm).view(B, T, n_kv, d_h).transpose(1,2)
V_self = (W_V^l · h_norm).view(B, T, n_kv, d_h).transpose(1,2)

# Apply RoPE to Q, K_self
Q, K_self = RoPE(Q, K_self, position_ids)

# GQA: repeat K_self, V_self to match n_h
K_self = repeat_kv(K_self, n_h // n_kv)       ∈ ℝ^{B × n_h × T × d_h}
V_self = repeat_kv(V_self, n_h // n_kv)
```

**Step 3: Bank injection** (if k ≥ 2 and bank.slots[l] non-empty)
```
h_bank = bank.slots[l]                        ∈ ℝ^{N_b × d}

# Project with learnable K-projector (residual)
h_bank_proj = h_bank + P^l(h_bank)            ∈ ℝ^{N_b × d}  (P^l: rank-r or full)

# Standard projection through frozen W_K, W_V (NO RoPE on bank)
K_bank = (W_K^l · h_bank_proj).view(N_b, n_kv, d_h).transpose(0,1)   ∈ ℝ^{n_kv × N_b × d_h}
V_bank = (W_V^l · h_bank_proj).view(N_b, n_kv, d_h).transpose(0,1)

# Add batch dim, repeat to n_h
K_bank = K_bank.unsqueeze(0).expand(B, n_kv, N_b, d_h)
V_bank = V_bank.unsqueeze(0).expand(B, n_kv, N_b, d_h)
K_bank = repeat_kv(K_bank, n_h // n_kv)      ∈ ℝ^{B × n_h × N_b × d_h}
V_bank = repeat_kv(V_bank, n_h // n_kv)

# Compute per-position bank gate
g = σ(W_gate^l · h_norm)                      ∈ ℝ^{B × T × 1}
```

**Step 4: Combined attention**
```
# Concat K/V along sequence dimension
K_all = cat([K_self, K_bank], dim=2)          ∈ ℝ^{B × n_h × (T + N_b) × d_h}
V_all = cat([V_self, V_bank], dim=2)

# Compute attention scores (with causal mask on self, no mask on bank)
scores = (Q @ K_all.transpose(-2,-1)) / √d_h  ∈ ℝ^{B × n_h × T × (T + N_b)}
mask_self = causal_mask(T)                    ∈ ℝ^{T × T}  (lower-triangular)
mask_bank = zeros(T, N_b)                     (bank fully visible)
mask_all = cat([mask_self, mask_bank], dim=-1)
scores = scores + mask_all                    (add -∞ to masked positions)

# Softmax
α = softmax(scores, dim=-1)                   ∈ ℝ^{B × n_h × T × (T + N_b)}

# Apply bank gate post-softmax (per-query on bank columns only)
g_expanded = g.unsqueeze(1).expand(B, n_h, T, 1)   ∈ ℝ^{B × n_h × T × 1}
gate_full = cat([ones(B, n_h, T, T), g_expanded.expand(B, n_h, T, N_b)], dim=-1)
α = α * gate_full

# Weighted sum over values
attn_out = (α @ V_all).transpose(1,2).reshape(B, T, d)   ∈ ℝ^{B × T × d}
h_attn = W_O^l · attn_out
```

**Step 5: Residual + MLP**
```
h_residual = h_{in}^{k,l} + h_attn
h_mlp = MLP^l(LayerNorm(h_residual))
h_out^{k,l} = h_residual + h_mlp              ∈ ℝ^{B × T × d}

# Override paused positions with input (skip-connection)
h_out^{k,l}[mask_skip] = h_{in}^{k,l}[mask_skip]
```

**Step 6: Halt decision** (after last layer, round k)
```
if l == L:
    p_halt^k = σ(W_halt · h_out^{k,L}[:, -1, :])   (last token, last layer)
    if p_halt^k > 0.5 or k == K_max:
        STOP (output h_out^{k,L})
    else:
        continue to round k+1 with h_in^{k+1,0} = h_out^{k,L} (or reset to h_out^{1,0})
```

### 7.3 Contrast with Alternatives

| Method | Key/Value Source | Projection | Trainable Params | Extra Output Tokens |
|---|---|---|---|---|
| **Standard Transformer** | K_self = W_K · h, V_self = W_V · h | Frozen W_K, W_V | 0 (inference) | 0 |
| **RAG** | K_self from prompt+retrieved_docs | Frozen W_K, W_V | External retriever (~100M) | 0 |
| **ALB (this work)** | K_self + K_bank (from bank.slots[l]) | K_bank = W_K · (I+P) · h_bank (P learnable) | P (~460K) + gates (~140K) | 0 |
| **LoRA** | K_self = (W_K + A·B) · h | W_K + low-rank adapter A·B | ~110M (r=64, all layers) | 0 |
| **ICL** | K_self from prompt+examples | Frozen W_K, W_V | 0 | 0 |
| **CoT** | K_self from prompt+generated_reasoning | Frozen W_K, W_V | 0 (or fine-tuned) | +10-100 (reasoning tokens) |
| **MEMIT** | K_self = (W_K + ΔW_K) · h | W_K edited (closed-form ΔW_K) | Edited weights | 0 |

**Unique to ALB**: External h-vector bank + learnable projector P + multi-round bank accumulation, all while keeping base W_K/W_V frozen.

---

## 8. Summary Table

| Comparison | Core Difference | ALB's Position |
|---|---|---|
| **vs RAG** | RAG injects *text*, ALB injects *hidden states* | ALB is "latent RAG" — no tokenization of memory |
| **vs LoRA** | LoRA edits *base weights*, ALB keeps *base frozen* | ALB is external memory, not weight adaptation |
| **vs ICL** | ICL puts examples in *prompt*, ALB in *bank* | ALB scales to large K without prompt explosion |
| **vs CoT** | CoT generates *reasoning tokens*, ALB does *reasoning rounds* | ALB is implicit pondering, not explicit text reasoning |
| **vs MEMIT** | MEMIT *edits W*, ALB stores in *external bank* | ALB is non-invasive, reversible, scalable |
| **vs Constitutional AI** | Constitutional AI is *objective*, ALB is *mechanism* | Orthogonal — ALB can serve any objective |

**Bottom line**: ALB is a novel memory mechanism that (i) augments attention with external hidden-state banks, (ii) uses a learnable projector to align bank content to layer-specific QK spaces, (iii) keeps base model weights frozen, (iv) supports both long-term (preloaded) and short-term (pause-write) memory, (v) operates via multi-round inference without generating extra output tokens.

**It is not** retrieval-augmented generation, not parameter-efficient fine-tuning, not in-context learning, not chain-of-thought, not model editing, and not a training objective. It is a **native attention-based memory mechanism** for frozen LLMs.

---

*Last updated: v2 inception (post-Exp42 B2 pivot)*
