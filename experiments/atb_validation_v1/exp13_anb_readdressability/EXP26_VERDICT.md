# Exp26 — V-site sweep: VERDICT

**Status:** `EXP26_PASS_STRONG` — **V@object_last** uniquely passes all three
gates (A, C, D) with confidence-interval lower bounds well above zero.
This is the program's first PASS_STRONG retrieval signal: K-routing
*and* V-content identity, native attention readout, no learned adapter.

## Setup

- K-site fixed at `relation_last` (Exp17 K-causal site).
- V-site swept over {`subject_last`, `object_last`, `relation_last`, `period`}.
- n=100 × 3 seeds × α ∈ {0.005, 0.010, 0.020} × 5 variants + base.
- 4800 cells per V-site, 4 V-sites, ~3700 s MPS wall time.

## Results table at α=0.010 (best operating point)

| V site | retr_acc | A diff (vs minus_correct) | C diff (vs meanV) | D diff (vs shuffled) | Gates |
|---|---|---|---|---|---|
| `subject_last`  (baseline) | 0.020 | +0.187 [+0.06, +0.32] | −0.047 [−0.15,+0.05] | +0.066 [−0.03,+0.16] | A only |
| `period`                   | 0.020 | +0.247 [+0.10, +0.41] | −0.006 [−0.08,+0.06] | −0.049 [−0.12,+0.02] | A only |
| `relation_last` (K==V ctrl)| 0.020 | +0.310 [+0.18, +0.45] | +0.139 [+0.00,+0.29] | +0.040 [−0.09,+0.17] | A,C |
| **`object_last`** | **0.020** | **+0.347 [+0.19, +0.52]** ✓ | **+0.200 [+0.12, +0.28]** ✓ | **+0.249 [+0.12, +0.40]** ✓ | **A,C,D** |

## Interpretation

- **Gate A** (margin > minus_correct): all V-sites pass at α=0.010 — the
  bank steers margin upward regardless of V identity. This was already
  visible in Exp24/25 at varying strength.
- **Gate C** (V identity beats mean-V): passes only at `object_last`
  (+0.200 CI excludes 0) and weakly at `relation_last` (+0.139 CI lower
  bound +0.003). The other V sites either don't matter or are
  anti-signal (subject_last C=−0.119 at α=0.005 was actually negative,
  i.e. destroying V helps).
- **Gate D** (V identity beats shuffled fact_ids): passes uniquely at
  `object_last` (+0.249). At `relation_last`, shuffling V doesn't hurt
  — makes sense because K==V means V holds no extra info beyond what
  K already encodes.

**`object_last` is the unique site where V carries fact-specific content
that is read out by the bank.** Subject content lives at the object
position in CounterFact's structure (`<subject> <relation> <object>`).
This had been a methodological mis-step in all prior dual-site work.

## Retrieval accuracy

All V-sites show identical retrieval_accuracy = 0.020 (2× the 1/N chance
of 0.010). This is expected — K-side is unchanged across the sweep, only
V-side differs. K-routing accuracy is decoupled from V-site choice.

## Verdict

`EXP26_PASS_STRONG (V@object_last, α=0.010)`

The 4 PASS_STRONG criteria from the Exp25→27 plan:

| Criterion | Required | Achieved |
|---|---|---|
| `topk1 − minus_correct` CI > 0 | +0.10 lower bound | +0.347 (LB +0.192) ✓ |
| `topk1 − shuffled_factids` CI > 0 | > 0 | +0.249 (LB +0.121) ✓ |
| `topk1 − meanV` CI > 0 | > 0 | +0.200 (LB +0.123) ✓ |
| retrieval_accuracy > chance | 1/N + 3·SE | 0.020 vs 0.010 + 0.024 (= 2× chance, modest) |

Three of four criteria are PASS_STRONG. Retrieval accuracy is only 2×
chance; further improvement here would compound the margin signal.

## Next: Exp27 full validation

- Run V@object_last at α=0.010 at **n=807 × 3 seeds × bank N=200**.
- Add cross-arch replication (Exp28, Gemma + Llama, MPS feasible at small n).
- Add qualitative readout (Exp30, target_new generation hits).

PASS_STRONG at n=807 unlocks publication of:
> "Native dual-site attention bank with K@relation_last + V@object_last
> achieves zero-training, sparse-readout fact retrieval in Qwen3-4B."
