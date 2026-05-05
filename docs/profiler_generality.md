# LOPI Profiler Cross-Architecture Generality

This document pins the architectural assumptions baked into
``deltamemory.memory.lopi_profiler.profile_residuals`` and the regression
tests that guard them
(``tests/test_profiler_generality.py``).

## What the profiler measures

For a HF causal LM with ``output_hidden_states=True``, the profiler
collects per-layer L2-norm statistics over the residual stream:

```text
hidden_states = (h_emb, h_1, h_2, ..., h_L)         # length L+1
norms_l[b, t] = ‖h_l[b, t]‖_2                       # for l in 1..L
mu_base[l]    = mean   over (b, t) of norms_l       # excludes padding
sigma_base[l] = std    over (b, t) of norms_l
mu_arch       = argmax_l sigma_base[l]              # ties: lowest index
eta_sigma     = 0.7  if  CV(sigma_base) > 0.5  else  1.0
```

The profile is *forward-only* (``torch.no_grad``, ``model.eval()``); LLM
weights are bit-equal pre/post.

## Architectural assumptions and their guards

| Assumption | Why it holds | Pinning test |
|---|---|---|
| Residual stream shape is ``(B, T, D)`` regardless of attention head structure (MHA / GQA / MQA) | All HF causal LMs sum the attention output back into the residual stream of width ``hidden_size``; ``num_heads`` and ``num_kv_heads`` are internal | `test_profiler_handles_dense_gqa_shape` |
| MoE residual is a sum over expert outputs and remains ``(B, T, D)`` | Mixtral / Qwen3-MoE / DeepSeek-V2 expose the post-routing residual via the same ``hidden_states`` tuple; the gate's ``argmax``-style top-k routing produces the same residual *shape* as a dense FFN | `test_profiler_handles_moe_routing` |
| Pre-attn-norm vs post-attn-norm convention is internal to the block | Llama uses pre-norm; Gemma3 / GPT-2 use post-norm — but ``hidden_states[l]`` is the *output* of block ``l`` either way, and that's what the profiler measures | `test_profiler_handles_post_norm_convention` |
| ``mu_arch`` is a dimensionless layer index in ``[0, L)`` | The argmax operates on the per-layer sigma array; no constant scaling by ``L``, no fixed-position assumption | `test_mu_arch_index_independent_of_layer_count` (L ∈ {2, 8, 24, 80}) |
| bf16 / fp16 model weights do not poison the float-domain statistics | ``_layer_norm_stats`` casts each ``h`` to ``float32`` before the L2 norm; the returned stats are plain Python floats | `test_profiler_runs_in_bfloat16_and_returns_float_stats` |
| The profile mutates no parameters across any architectural variant | ``model.eval()`` + ``torch.no_grad()`` + no backward pass; the dataclass return type holds plain Python floats only | `test_profiler_weights_bit_equal_across_archs` (dense / MoE / post-norm) |
| Padding tokens are excluded from the (B, T) population | ``_layer_norm_stats`` masks out ``attention_mask == 0`` positions before computing mean/std | `test_layer_norm_stats_excludes_padding_tokens` (existing) |

## What the profiler does NOT depend on

These are *non*-assumptions: changing them does not break the profiler.

* **Number of attention heads** — the residual stream has already
  contracted across the head dimension by the time it returns.
* **Head dimension** — same reason; ``hidden_size = num_heads *
  head_dim`` is the only width the profiler sees.
* **RoPE backend** (Llama / Gemma2 / Gemma3 / Qwen3 / GLM-4 partial)
  — RoPE is internal to attention; the residual is the sum of the
  RoPE'd attention output back into the un-rotated residual stream.
* **Tokenizer family** — the profiler accepts any HF tokenizer that
  supports ``return_tensors='pt'`` and ``padding=True``.
* **Tied vs untied output embedding** — irrelevant to mid-stack
  residuals.
* **KV-cache format** — ``use_cache=False`` is forced, so the profile
  is invariant to the cache scheme.

## Adding a new architecture

The profiler should "just work" on any HF causal LM that returns the
standard ``hidden_states`` tuple from ``output_hidden_states=True``.  If
a new architecture deviates:

1. **Returns ``hidden_states=None``**: ``profile_residuals`` raises a
   clear ``RuntimeError`` (``lopi_profiler.py:220``).  Add an explicit
   wrapper before opening an issue.
2. **Different ``attention_mask`` convention** (e.g. 4D float mask):
   ``_layer_norm_stats`` validates the mask is shape-compatible with the
   per-layer norm tensor; mismatches raise ``ValueError``.
3. **Encoder-decoder or vision-language**: out of scope.  The U-LOPI
   stack is causal-LM-only.

When adding a new family, copy the existing pattern from
``test_profiler_generality.py``: write a minimal ``_FakeBlock`` whose
``forward`` returns ``residual_in + linear(residual_in)`` and add it to
``_ConfigurableModel``.  No real HF download is required.

## Calibration heuristics (orthogonal to architecture)

* ``eta_sigma`` defaults to ``1.0`` and switches to ``0.7`` when the
  cross-layer coefficient of variation of ``sigma_base`` exceeds
  ``0.5``.  This is a *tunable* (D-S-6 in the Phase S plan), not an
  architectural assumption.  If a future architecture has a flatter
  sigma profile, the threshold may want re-tuning, but the
  *architectural* contract still holds.
* The default profile corpus is N=10 short multilingual prompts
  (``DEFAULT_PROFILE_CORPUS``).  Callers can override via the
  ``prompts`` kwarg without changing the calibration math.

## Cross-link

* The position-invariance counterpart for the bank scoring path is
  documented in ``docs/rope_invariants.md``.
* Implementation: ``deltamemory/memory/lopi_profiler.py``
* Module-level docstring: ``deltamemory/memory/lopi.py`` (now corrected
  to remove the stale "configurable" pre/post-RoPE Q claim).
