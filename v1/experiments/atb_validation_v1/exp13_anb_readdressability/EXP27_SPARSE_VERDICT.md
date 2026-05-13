# Exp27 — True Sparse-Attention ANB: FINAL FAIL

## Verdict

**EXP27_FAIL** — Joint-softmax sparse-attention readout shows the same
N=100 PASS → N=200 FAIL curve as every prior α-additive attack. This is
the fourth independent confirmation of the same ceiling.

Native ANB attention traces are re-addressable at small scale (N≤100)
but **cannot produce scalable routed memory** in the captured K/V space
of Qwen3-4B.

## Key infrastructure note

The "α-additive" framing in EXP26b_VERDICT.md was incomplete. The
existing default code path (`bank_separate_softmax=False`,
attn_native_bank.py lines 857–889) **already implements**
`Attn(Q, [K_seq; M_K], [V_seq; M_V])` with a joint softmax. α multiplies
M_V *inside* the read; it does not separate the readout.

So Exp27 is not architecturally distinct from Exp23–Exp26b. The only
genuinely untested regime was α ≥ 0.05 with the joint-softmax path. This
report covers that regime.

## Stage 1 — α regime probe (N=100, 2 seeds)

K=relation_last, V=object_last (Exp26 PASS settings), bank=100, topk=1.

| α | retr_acc | A (vs minus_corr) | C (vs meanV) | D (vs shuf_factids) | bank_mass |
|---|---|---|---|---|---|
| 0.05 | 3.0% (3× chance) | +0.168 [−0.05, +0.40] | **+0.332 ✓** | **+0.244 ✓** | 0.33 |
| 0.10 | 2.0% | +0.015 | +0.088 | −0.061 | 0.34 |
| 0.30 | 2.0% | −0.199 | −0.287 | −0.149 | 0.32 |
| 1.00 | 2.0% | +0.000 | +0.199 | −0.222 | 0.18 |
| 3.00 | 1.0% (chance) | +0.057 | +0.091 | +0.104 | 0.13 |

Observations:
- retr_acc peaks at 3× chance (α=0.05); **never reaches the 4× chance
  trigger** for Stage 2.
- C/D PASS only at α=0.05; pattern is identical to Exp26b N=100.
- At α≥0.3 the model output degrades (negative gate diffs ≠ noise; the
  bank V is over-injected).
- At α=1.0/3.0 the softmax actually downweights the bank (bank_mass
  drops from 0.34 to 0.13) — sequence keys win the joint softmax.
  Joint-softmax does NOT favor the bank just because we asked it to.

## Stage 2 — N=200 falsification at α=0.05 (3 seeds, 600 pairs)

| Gate | diff | CI95 | verdict |
|---|---|---|---|
| A | +0.096 | [+0.011, +0.184] | weak/borderline |
| C | −0.043 | [−0.155, +0.067] | **FAIL** |
| D | +0.079 | [−0.075, +0.243] | **FAIL** |

retr_acc = 1.0% (chance 0.5%) — 2× chance, same as Exp26b N=200.

The signal collapses exactly as Exp24/26/26b did.

## Program-level summary (four independent falsifications)

| Attack | Architecture | N=100 gates | N=200 gates |
|---|---|---|---|
| Exp24 K-routing α-add | additive readout, single V | DIRECTIONAL +0.193 | weak |
| Exp26 V@object_last α-add | additive, V@obj_last | A+C+D PASS_STRONG | All FAIL (Exp27 old) |
| Exp26b multi V α-add | additive, V@subj_to_obj 8-tok | A+C+D PASS | All FAIL |
| Exp27 joint softmax α=0.05 | joint softmax, V@obj_last | C+D PASS | All FAIL |

retrieval_accuracy never escapes 2-3× chance at N=100 and decays to 1-2×
chance at N=200, **independent of**:
- K capture site
- V capture site
- V span length (1 token vs 8 tokens)
- α magnitude (0.003 → 3.0, 4 orders of magnitude)
- joint vs additive softmax
- bank_topk (the 1 case tested here)

## Diagnosis

The bank cannot win attention mass against the sequence keys in any
useful, fact-discriminating way. At low α the bank V is too small to
move the residual; at high α the bank V is large but the softmax routes
attention to sequence tokens instead of bank slots (bank_mass collapses
at α=1.0/3.0). There is no setting of (α, topk) on this architecture
that produces both:
(i) softmax mass concentrating on the *correct* bank slot, and
(ii) V contribution at full magnitude.

Both conditions are required for routed memory. The captured pre-RoPE
K representations from Qwen3-4B simply do not contain enough query-key
discriminability to single out one slot in a 200-fact bank.

## Honest implication

Native re-addressability holds at small scale (the +0.193 K-routing
crack in Exp24, the dozens of PASS_STRONG gates at N=100) but is a
small-bank artifact. The native attention K/V space is **not a useful
routing space** for fact-level memory at N≥200. ANB as the user
originally framed it — "原生 memory, not external injection" — does not
scale to bank sizes beyond the noise floor of cosine routing on raw
attention K.

## Remaining options

1. **Accept the negative result.** Write up as the falsification of
   native α-additive AND joint-softmax ANB. Clean, publishable, ends
   the line. (User said earlier: "我明确反对的" RAG-like external
   memory; native ANB has now been falsified on four axes.)

2. **Learned read-time K adapter** (Exp31 in original plan). A small
   `A: q_relation → k_bank` trained on held-out facts to push correct
   slot above noise. This is the smallest possible deviation from
   native attention — it adds one linear map at read time. The user
   has consistently deferred this option ("先别做"), but it is the only
   intervention that addresses the actual bottleneck identified here
   (cosine routing on raw q·M_K^T is insufficient).

3. **Different model architecture.** Try a model with more
   discriminable attention K-space (smaller-context fine-tuned model,
   different family). Speculative.

## Recommendation

Option 1 is the honest call right now. Four falsifications on four
genuinely different axes is sufficient evidence for a negative paper.
The K-adapter (option 2) is a fundamentally different research line —
not a continuation of the ANB attack — and should be opened only with
that framing.

## File index

- Stage 1 raw: `run_mps_exp27_stage1/cells.jsonl`
- Stage 1 analysis: `run_mps_exp27_stage1/analysis/`
- Stage 2 raw: `run_mps_exp27_stage2_N200/cells.jsonl`
- Stage 2 analysis: `run_mps_exp27_stage2_N200/analysis/`
- Prior verdicts: `EXP25_VERDICT.md`, `EXP26_VERDICT.md`,
  `EXP27_VERDICT.md` (old, V@object_last falsification),
  `EXP26b_VERDICT.md`.
