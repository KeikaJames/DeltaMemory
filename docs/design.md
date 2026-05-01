# Delta Memory Design

Delta Memory is a storage-backed, layerwise attention-memory mechanism for
frozen Gemma-style decoder language models. The current Python package remains
`rcvhc` for compatibility.

The main path is not prompt retrieval. Retrieved `source_text` is debug/citation
metadata only. It is never appended to the prompt in the Delta Memory path.

## Symbols

- `N`: total historical tokens ingested.
- `W`: local window tokens resident for the current question.
- `B`: memory block size.
- `K`: retrieved memory blocks per query.
- `L_e`: enabled memory injection layers.
- `H`: number of attention heads.
- `d_h`: attention head dimension.
- `d_m`: compressed memory dimension.

## Storage-Bounded Context Claim

Delta Memory targets storage-bounded context, not infinite context.

GPU resident memory:

```text
M_gpu ~= M_model + M_local_KV(W) + M_retrieved(K, L_e) + O(1)
```

where:

```text
M_local_KV(W) = O(W * L * H * d_h * bytes)
M_retrieved(K, L_e) = O(K * L_e * d_mem_attn * bytes)
```

External storage:

```text
M_store(N) = ceil(N / B) * S_block
S_block ~= bytes(K_raw + V_raw + Delta_q + Delta_k + Delta_v + metadata)
```

Usable history length is bounded by physical storage, retrieval latency, and
memory compression rate. It is not bounded by keeping the full history in the
GPU context window.

## Memory Write

For layer `l` and source block `b`:

```text
c_self_b^l = Pool(H_b^{l,in}, H_b^{l,out})
U_{b,i}^l = normalized forward usage from future tokens i to block b
v2_b^l = sum_i U_{b,i}^l * c_use_i^l
Delta_b^l = RMSNorm(v2_b^l - c_self_b^l)
```

Each memory block stores:

```text
M_b^l = {
  raw_key_b^l,
  raw_value_b^l,
  delta_q_b^l,
  delta_k_b^l,
  delta_v_b^l,
  usage_mass_b^l,
  metadata
}
```

The P0 writer uses frozen model hidden states and attentions, then writes the
compressed tensors to CPU storage.

## Retrieval

For a question, the engine computes a read query from the local window and
retrieves top-k memory blocks for each enabled attention layer:

```text
R_l = topk(q_read, MemoryStore_l)
```

The top-k tensors are the only external memory tensors moved to the model
device. The full store remains on CPU or disk.

## Attention Injection

For each enabled attention layer `l`:

```text
q, k, v = GemmaSelfAttentionProj(h_t^l)
dq = P_q(Delta_b^l)
dk = P_k(Delta_b^l)
dv = P_v(Delta_b^l)
```

The Q/K/V residual mode is:

```text
q' = q + alpha_q * gate_q * dq
k' = k + alpha_k * gate_k * dk
v' = v + alpha_v * gate_v * dv
```

The memory-prefix attention form is also compatible with this definition:

```text
K_ext = concat(K_local, K_mem + gate_k * dk)
V_ext = concat(V_local, V_mem + gate_v * dv)
Attention(Q', K_ext, V_ext)
```

The implementation hooks Q/K/V projection modules and adds these residuals
directly inside every enabled attention layer. Single-layer injection is an
ablation, not the main path.

## Main and Baseline Modes

Main research modes:

- `delta_v`
- `delta_qv`
- `delta_kv`
- `delta_qkv`

Controls:

- `delta_qv_zero`
- `delta_qv_random`
- `delta_qv_shuffled`
- `delta_qv_force_gate`

Baselines:

- `no_memory`
- `raw_memory`

`raw_memory` is a compressed-memory baseline. RAG prompt insertion is not part
of the main path and must be implemented only as a separate baseline if needed.

## Trace Requirements

Every answer run reports:

- answer metrics: NLL, rank, top-k, log probability.
- retrieved memory IDs and scores.
- token ranges and source snippets for debug only.
- Q/K/V delta norms and gate statistics.
- memory count and storage bytes.
- trainable base parameter count, expected to be zero.

## Scientific Claim Boundary

Engineering success means the store, retrieval, frozen Gemma forward, Q/K/V
hooks, and controls run without training base weights.

Scientific signal requires aligned Delta to beat zero, random, and shuffled
controls. CPU-only or tiny smoke runs must be described as wiring-only unless
the controls support a stronger claim.

## Optimization Boundary

The frozen base model is not expected to know how to use Delta Memory without
training. The trainable Delta Memory surface is deliberately small:

- writer projections that compress frozen hidden/attention states into memory,
- Delta-to-Q/K/V projection matrices,
- gate biases and gate weights,
- optional readout baselines.

Training minimizes answer-token loss under memory-enabled conditions while
keeping every Gemma parameter frozen. Valid training evidence must compare the
trained memory path against zero, random, shuffled, no-memory, and raw-memory
controls.
