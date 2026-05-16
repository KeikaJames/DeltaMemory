# e10 — Top-K retrieval over the AttentionBank

## Research question

e11 wave-3 found that the trained rank-64 K-projector + **any** non-empty bank
(real, iid Gaussian L2=15, single replicated row, constant vector) gives a
large NLL drop of similar magnitude. Bank content barely matters. That makes
the "associative memory" story look hollow: the so-called memory may just be
a content-agnostic capacity bump from the trained projection.

e10 tests a falsifier:

> Does **content-sensitive top-K retrieval** over a real bank produce a
> clearly larger NLL improvement than top-K over a random bank, or than
> all-attend over a random bank?

If yes → retrieval is doing real work. If no → capacity hypothesis stands.

## Variants

| `--variant`                  | Bank pool                  | Selection rule                  | Notes |
|------------------------------|----------------------------|---------------------------------|-------|
| `topk_cosine_real_K1`        | real (Exp35b b-vectors)    | top-1 by cosine to query        | smallest K |
| `topk_cosine_real_K8`        | real                       | top-8 by cosine                 | primary signal variant |
| `topk_cosine_real_K64`       | real                       | top-64 by cosine                | matches projector rank |
| `topk_cosine_random_K8`      | random iid Gaussian L2=15  | top-8 by cosine                 | should be flat if retrieval is content-sensitive |
| `topk_random_indices_K8`     | real                       | 8 random row indices (no cosine) | should be flat if retrieval is content-sensitive |
| `all_attend_real`            | real                       | all N entries                   | canonical e01 baseline |
| `all_attend_random_renorm15` | random iid Gaussian L2=15  | all N entries                   | matches e11/n1 baseline |

All variants share: Qwen3-4B, MPS, bf16, layer 9, rank-64 K-projector + bank
gate heads as the only trainable params, lr 2e-4, 200 AdamW steps, gradient
clip 1.0, `n_train=120`, `n_eval=120`, `n_preload=512`, seed 0.

## Query (for cosine retrieval)

Default `--query_mode mean_embed`: query = mean of the input token embeddings
(one fast lookup, no extra forward pass). The exact value used is recorded in
the output JSON under `query_mode`.

Alternative `--query_mode last_hidden`: extra no-bank forward, query = last-token
hidden state at layer `bank_layer`'s input. More principled but ~2× cost.

## Implementation

Retrieval is applied **before** the projector: for each forward, we pick a
subset of `b_pool_full` (real or random), pass it through the trainable
projector `P` (residual `x + P(x)`), and write the projected K rows into
`bank.slots[bank_layer]`. The unmodified attention patch then attends over
exactly those K bank entries. No 2-pass forward needed in the default mode.

## Pass criterion

A single run only records its own `delta_real = post_real - base`. The
**cross-variant** pass requires two runs:

1. `Δ_real(topk_cosine_real_K8) ≤ −1.0`
2. `post_real(all_attend_random_renorm15) ≥ post_real(topk_cosine_real_K8) + 1.0`

If both hold, content-sensitive retrieval beats random-bank capacity by a
clear margin. Otherwise the capacity hypothesis wins.

## Run (do not launch on MPS while wave5 is occupying it)

```
python v2/experiments/e10_topk_retrieval/run.py --variant topk_cosine_real_K8
python v2/experiments/e10_topk_retrieval/run.py --variant all_attend_random_renorm15
# ... etc
```

Outputs land at `v2/experiments/e10_topk_retrieval/e10_<variant>_seed<seed>.json`.
