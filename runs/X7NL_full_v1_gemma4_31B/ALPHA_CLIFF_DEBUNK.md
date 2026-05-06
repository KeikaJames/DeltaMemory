# α=0.25 "cliff" debunk — gemma-4-31B-it X.7-NL subB

**TL;DR**: There is no α=0.25 phase transition. The apparent "cliff"
in the original v0.6 subB sweep was bf16 + seed noise. After running
the same grid with 3 seeds we see a high-variance bowl in α∈[0.05, 0.50]
and a low-variance plateau in α≥0.70.

## Evidence

| α range          | n_seeds | mean log_margin | per-seed std |
|------------------|---------|-----------------|--------------|
| [0.05, 0.50]     | 3       | varies         | ≈ 1.70 nats  |
| [0.70, 2.00]     | 3       | +0.365         | ≈ 0.31 nats  |

The std in the noisy bowl is ~5× the std on the plateau. Single-seed
runs in v0.6 happened to land in the bowl trough at α=0.25 and were
mistakenly read as a phase transition.

## Why no transition is mathematically possible

α enters the bank-V output as a multiplicative scalar. It is **outside**
all softmaxes; the post-attention residual is

    h' = h + Δ_attn(x) + α · BankReadV(x)

where `BankReadV(x)` is itself a softmax-weighted aggregation. For
`BankReadV` to undergo a phase transition at α=0.25, the softmax
temperature would have to depend on α — it does not.

## Implication for `safe_alpha.py`

The v0.6 module modelled an α=0.25 "cliff" using single-seed residual
norms. That model is invalid. The v0.7 rewrite (this commit) replaces
it with:

* `empirical_alpha_sweep(probe_fn, alphas, seeds, …)` — multi-seed
  sweep that reports mean / std / 95% CI per α and flags the high-
  variance region.
* `SafeAlphaScheduler(deploy_floor=0.7, recommended=1.0, …)` — keeps
  α below the noise floor out of production by mapping it up to a
  deployable α (default 1.0).
* `recommend_alpha_from_sweep(sweep, margin_floor, noise_std_budget)`
  — returns the smallest α that simultaneously clears a margin floor
  and stays under a noise budget.

Legacy names (`compute_safe_alpha_threshold`, `validate_scheduler_vs_naive`,
`CLIFF_*_DEFAULT`, `POST_CLIFF_ALPHA_DEFAULT`) remain as deprecated
shims so PR #38 imports keep working.

## Reproducibility

The actual subB α-sweep cells live in `cells.jsonl` (one JSON line per
(α, seed) cell). Compute per-α std with:

```python
import json, statistics, collections
buckets = collections.defaultdict(list)
for line in open("cells.jsonl"):
    c = json.loads(line)
    buckets[round(c["alpha"], 3)].append(c["log_margin"])
for a, vs in sorted(buckets.items()):
    if len(vs) >= 2:
        print(a, round(statistics.fmean(vs), 3), round(statistics.stdev(vs), 3))
```

If this script ever shows std < 0.5 in the [0.05, 0.50] range with
≥3 seeds, this debunk is wrong and we should reopen the cliff hypothesis.
Until then: **deploy with α≥0.7 and stop modelling phantom cliffs.**
