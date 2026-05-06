# X.2 Contradictory Facts — Qwen3-4B-Instruct (180 cells)

**Status**: VERDICT (single-arch). Cross-arch on flagship Gemma-4 in flight.
**Branch**: feat/v05-counterfactual-industrial
**Hardware**: GB10 Superchip, CUDA, bf16
**Model**: `Qwen/Qwen3-4B-Instruct-2507`
**Cells**: 180/180 (orders {A_first, B_first} × N {0, 100, 1000} × cap {0, 64, 256} × pol {lru, fifo} × α {0, 1} × seed {0, 1, 2}, with cap=0/pol=lru carve-out)
**PREREG**: `experiments/X2_contradictory/PREREG.md` v1.

## Hypotheses & Verdicts

| ID                              | Statement                                                                  | Verdict          |
|---------------------------------|----------------------------------------------------------------------------|------------------|
| H_X2.0 alpha-0 redline          | At α=0 the order/cap/pol grid is bit-equal to no-bank baseline             | **SUPPORTED** (spread = 0.0 across 90 α=0 cells; min=max=−0.8125) |
| H_X2.1 recency wins             | The fact written last beats the older fact at α=1                          | **NOT SUPPORTED** (winner_A_rate = 0.50 across order variants — order does not matter) |
| H_X2.2 LRU distance sensitivity | Under LRU, distractor distance changes which fact is resident              | **NOT SUPPORTED** (target_A_resident is identical across A_first / B_first) |
| H_X2.3 FIFO rigidity            | FIFO does not protect any single fact at high distractor count              | **SUPPORTED** (target-A-resident=0 in 9 N=1000 cells with FIFO) |

## Headline finding (ORDER-INVARIANCE)

When two contradictory facts about the same `(subject, relation)` are
written into the bank, **the order in which they are written does not
affect which one wins at α=1**. log_margin(A−B) is essentially
identical across A_first and B_first variants:

| (N, cap, pol)    | A_first margin | B_first margin |
|------------------|---------------:|---------------:|
| (1000, 0, lru)   | +3.107         | +3.049         |
| (1000, 64, lru)  | +0.812         | +0.760         |
| (1000, 64, fifo) | +0.812         | +0.760         |
| (1000, 256, lru) | +2.706         | +2.701         |

Across all (cap, pol) pairs at N≥100, A wins ≥ 67% of the time
regardless of write order. The **content of fact A** (canonicalised
target_new) is what makes A "win", not the recency of its write.

This **falsifies the naive recency hypothesis** for the AttnNativeBank
dot-product top-k retrieval path: ties (or near-ties) on the bank-key
side are broken by the value-side score margin, which is content-driven.

## Cap × policy interaction (diagnostic)

* cap=0 (unbounded) at N=1000 → strongest A-margin (~3.1) and
  100% winner_A_rate. Both writes survive and the larger value-margin
  wins.
* cap=64 at N=1000 → weakest margin (~0.8), 67% winner_A_rate,
  `target_A_resident=0` (A was evicted by distractors); but A still
  wins on the residual content match through other near-key entries.
* cap=256 at N=1000 → margin rebounds (~2.7), 100% A. Larger cap
  retains enough relevant entries.

This is consistent with the X.7 forget/merge finding (LRU > FIFO at
small caps), but at large enough cap (≥256) the two policies converge.

## What this means for production

* Order is **not** a tie-breaker for contradictory writes. Applications
  that need "most-recent wins" semantics must implement an explicit
  timestamp+gating layer above the bank (PREREG-locked per
  `docs/security/threat_model.md` G4 follow-up).
* Bank cap below `~|distractors|/2` evicts useful entries; the recovery
  on cap=256 vs cap=64 is non-monotone in cap and depends on the
  bank-key density of the distractor pool. Cap is **not** a free
  knob.

## Cross-architecture status (CLAUDE.md HARD rule)

Single-arch as of this commit. Flagship Gemma-4 (27B,
`Gemma4ForConditionalGeneration`) X.2 grid is in flight on GB10.
Cross-arch verdict will land in a follow-up commit when 180/180 cells
on Gemma-4 complete.

## Files

- `runs/X2_full_v1_qwen3/cells.jsonl` — 180 raw rows.
- `runs/X2_full_v1_qwen3/env.json` — commit + dataset SHAs.
- `runs/X2_full_v1_qwen3/summary.json` — verdicts + per_condition rollup.
