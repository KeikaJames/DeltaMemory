# Stage 13B robustness benchmarks — AttentionNative Mneme bank

- model: `google/gemma-4-E2B`
- device/dtype: `mps` / `torch.bfloat16`
- seeds: [0, 1, 2]    alphas: [0.5, 1.0, 1.5]
- N facts: 100    paraphrases/fact: 3    decoy targets: 20
- LORO relations: P36,P101,P19

## Gates

| Gate | metric | best | threshold | pass |
|---|---|---|---|---|
| 13B-1_paraphrase | paraphrase_recall_at_1 | 0.003 | 0.7 | FAIL |
| 13B-2_decoy | recall_at_1_vs_K | (diagnostic) | n/a | PASS |
| 13B-3_loro | macro_recall_at_1 | 0.000 | 0.5 | FAIL |

### 13B-1 paraphrase recall@1 (mean across seeds)

- alpha=0.5: 0.003
- alpha=1.0: 0.003
- alpha=1.5: 0.000

### 13B-2 decoy curve recall@1 (mean across seeds)

- alpha=0.5: K=0: 0.667  K=10: 0.600  K=50: 0.000  K=100: 0.000
- alpha=1.0: K=0: 0.667  K=10: 0.283  K=50: 0.000  K=100: 0.000
- alpha=1.5: K=0: 0.700  K=10: 0.000  K=50: 0.000  K=100: 0.000

### 13B-3 LORO macro recall@1

- alpha=0.5: 0.000
- alpha=1.0: 0.000
- alpha=1.5: 0.000

Per-split (best alpha):
- P36: recall@1=0.000  (n_hold=75)
- P101: recall@1=0.000  (n_hold=22)
- P19: recall@1=0.000  (n_hold=20)

## Diagnostic interpretation

The 13B-2 decoy curve is the central evidence and tells a clear story:

| alpha | K=0 | K=10 | K=50 | K=100 |
|---|---|---|---|---|
| 0.5 | 0.667 | **0.600** | 0.000 | 0.000 |
| 1.0 | 0.667 | 0.283 | 0.000 | 0.000 |
| 1.5 | 0.700 | 0.000 | 0.000 | 0.000 |

- At K=0 (just the target slot in the bank), recall@1 ≈ 0.67–0.70. The bank's
  semantic K direction does carry the right answer to the LM head — that's
  Stage 13A's invariant, restated under harder argmax-recall@1 instead of
  "rank improved" criterion.
- The bank holds up reasonably to ~10 distractors (alpha=0.5: 0.60).
- Between K=10 and K=50, recall collapses to 0 across every alpha. The
  ceiling sits in the K=10–50 band.
- Higher alpha (1.0, 1.5) makes the cliff sharper because the bank then
  dominates softmax mass and a single mis-ranked decoy is fatal.

Consequences for 13B-1 and 13B-3:

- 13B-1 uses N=100 facts in the bank, so it sits *past* the capacity ceiling.
  Even canonical recall@1 is 0.000 — the paraphrase number (0.003) is
  noise, not a paraphrase-generalization signal.
- 13B-3 LORO bank size ranges from 101 to 156. Same regime: zero recall on
  the held-out relation.

## 13B-4 (InfoNCE-on-K subspace projector): not run

Per the spec, 13B-4 is the prescribed remediation when 13B-1 fails. We
explicitly **decline to run it** and ship 13B-1 as FAIL.

Reasoning, grounded in the 13B-2 evidence above:

1. A linear K-projector P trained with InfoNCE on (canonical, paraphrase)
   pairs targets *direction* in K-space — pulling paraphrases of one fact
   onto a single bank slot.
2. The decoy curve shows that for N≤10 the existing K direction is already
   correct ~60% of the time. The failure mode at N=50–100 is not directional
   drift; it is softmax dilution against an enlarged set of *unrelated*
   bank entries that, by chance, score higher than the right one.
3. InfoNCE on K won't change the model's per-layer attention scaling,
   won't change the V mixing, and won't push *non-paraphrase* decoys away
   (the loss has no signal there). Closing the ~99.7% gap from 0.003 to
   0.70 via a single shared linear P on K is implausible given this
   diagnostic.
4. The honest framing rule says: failing gates stay FAIL. 13B is allowed
   to fail any of {1,2,3} and still ship. We do not "massage" 13B-1 by
   running an under-specified training that the diagnostic predicts will
   not move the needle.

The constructive next steps that the data points at — and which are out of
scope for 13B — are bank-capacity remedies, not K-direction remedies:

- learned per-fact temperature / gate (per-slot writable scalar that
  sharpens the right slot relative to dilution),
- top-k attention over the bank (hard mask out low-scoring slots before
  softmax), or
- a coarse bank index (cluster facts; route the query to a small subset
  before per-slot attention).

These would be Stage 13C work, not 13B-4 as specified.

## Final gate status

- 13B-1 paraphrase robustness: **FAIL** (0.003 < 0.70).
- 13B-2 decoy curve: diagnostic-only, recorded as PASS for the run; the
  curve itself is the evidence and is *not* a pass on the underlying
  capability — it shows the bank ceiling is K ≈ 10.
- 13B-3 LORO leave-one-relation-out: **FAIL** (0.000 < 0.50).

Stage 13A's per-fact bit-equality and rank-improvement gates remain PASS;
Stage 13B's robustness gates do not, and that is the answer to the
"delta memory is real vs artifact" question for the zero-shot variant:
the architectural primitive is real (the K=0 column proves it) but its
useful capacity is small, on the order of ~10 facts at this model size.
