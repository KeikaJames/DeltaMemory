# X.1 Bank-Scaling Dilution — Flagship Verdict (gemma-3-1b-it MPS bf16)

**Status**: NOT SUPPORTED across the board, in an informative way.
**PREREG**: `experiments/X1_bank_scaling/PREREG.md` (X.1.v1).
**Cells**: 72 / 72 (`runs/X1_full_v2/cells.jsonl`).
**Grid**: N ∈ {1, 10, 100, 1000} × arms ∈ {none, topk_4, separate_softmax} × α ∈ {0, 1} × 3 seeds.

## Headline log-margin curves (α=1, mean over 3 seeds)

| arm \ N | 1 | 10 | 100 | 1000 |
|---|---:|---:|---:|---:|
| `none` | **-11.375** | -1.000 | -1.406 | **+0.247** |
| `topk_4` | -11.375 | -3.958 | -3.958 | -2.385 |
| `separate_softmax` | **+1.500** | -1.146 | -1.479 | -1.375 |

α=0 redline = -5.000 constant across every (arm, N, seed) cell. α-shield holds bit-equal at flagship scale ✅.

## Hypothesis verdicts

| H | Claim | Verdict | Mechanism |
|---|---|---|---|
| H_X1.1 | `none` arm shows monotone dilution, ≥2× drop by N=100 vs N=1 | **NOT SUPPORTED** | At flagship scale `none` is **non-monotone** and ends *higher* (+0.247) than it started (-11.375). Naive softmax dilution prediction is falsified. |
| H_X1.2 | `bank_topk=4` keeps margin within 20% of N=1 across all N | **NOT SUPPORTED** | max relative deviation = 0.79 (target < 0.20). topk does *constrain* drift but not at the prereg threshold. |
| H_X1.3 | `bank_separate_softmax` keeps margin within 10% of N=1 across all N | **NOT SUPPORTED** | max rel dev = 1.99. separate_softmax is the only arm with positive margin at N=1, but degrades by N=10. |

## Interpretation (疑问① user concern)

User asked: "随着 bank 增大，Softmax 分母爆炸 → 早期重要记忆的权重被严重稀释 …
即使有 LOPI 门控，长期来看是否会出现'记忆被自己埋葬'的现象？"

**At flagship gemma-3-1b-it scale, the dilution-by-埋葬 mechanism is NOT
the dominant failure mode**. The data show:

1. The `none` (no defense) arm produces its **worst** margin at N=1, not
   N=1000. At N=1 the bank has a single entry whose retrieval is
   undermined by the post-RoPE query mismatch; as N grows, in-distribution
   distractor entries appear to *anchor* the attention distribution and
   the target rises with them.
2. `bank_topk=4` does what the user expected — it caps growth — but its
   ceiling at this scale (≈ -2.4 at N=1000) is **worse** than `none`'s
   N=1000 (+0.247). Defense is over-zealous.
3. `bank_separate_softmax` wins at N=1 (+1.500) and is the only
   defender that produces a *positive* margin, but it does not preserve
   that margin at scale.

This means the X.7 lifecycle (LRU forget/merge) **must** be evaluated
against a non-monotone baseline: a capacity cap that throws out late
entries could *hurt*, not help, if late entries are the anchoring ones.

## Caveats

- Single model (gemma-3-1b-it). Tier B on a larger flagship (cached on
  GB10: DeepSeek-R1-Distill-Qwen-32B, Qwen3.5-35B-A3B-Base) is the
  obvious follow-up to confirm the non-monotonicity is scale-dependent
  vs architecture-dependent.
- Single distractor pack. Distribution shift could invert the curve;
  Phase X.6 will exercise that.
- log_margin floor of -5.000 is the metric clamp; the N=1 row at
  -11.375 indicates margin floored even further (clamp may need to
  loosen for flagship-scale).

## Authenticity

- commit: `b70463c3` (this branch); will pin in env.json on PR.
- device: mps, dtype: bf16.
- α=0 drift-zero witness: 36/36 cells return -5.000 ✅.
- Raw cells preserved at `runs/X1_full_v2/cells.jsonl` (72 lines, JSONL).
- summary.json is a pure function of cells.jsonl (`aggregate.py`).
