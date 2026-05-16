# E20c Verdict — Adversarial Audit Falsifies e20b "north-star"

**Status**: 🔴 e20b PASS retracted. The 4.96-nat "Δ_A_init" was a global
style attractor, not item-specific memory.
**Date**: 2026-05-16
**Model**: Qwen/Qwen3-4B-Instruct-2507

## a. What e20b claimed and what e20c tested

e20b: with frozen projector + trainable b_A vectors, three seeds reached
Δ_A_init ≈ 4.96 nat with Δ_A_after_evict ≈ 0 — declared a north-star PASS
for item-specific memory.

e20c re-runs the same protocol on seed 0 with one structural addition:
- setA split into trainA (256) and heldA (256). Only trainA items provide
  gradient.
- After training, run four extra evaluations:
  1. Shuffle b_A row indices in the bank (preserve content, break alignment).
  2. heldA items with identity-bank (b_A in original order).
  3. Drift items (64 random keys disjoint from setA ∪ setB) with bank.
  4. Cross-checks already in e20b (b_B, empty bank).

If lift is item-specific, shuffling should crash it; held-out should be much
lower than trainA; drift items should be unmoved.

## b. Results (seed 0)

| eval condition | NLL | Δ vs baseline |
|---|---:|---:|
| baseline trainA | 12.16 | 0 |
| baseline heldA  | 12.13 | 0 |
| baseline drift  | 12.28 | 0 |
| trainA, identity b_A | 7.56 | **+4.60** |
| trainA, shuffled b_A | 7.56 | **+4.60** |
| heldA, identity b_A  | 7.99 | **+4.14** |
| heldA, shuffled b_A  | 7.99 | **+4.14** |
| drift, b_A installed | 8.06 | **+4.22** |
| trainA, b_B evicted  | 12.16 | −0.001 |
| setB, b_B            | 11.93 | −0.011 |
| trainA, empty bank   | 12.16 | 0 |

## c. Killer numbers

- **Shuffle Δ – Identity Δ = +0.005 nat.** Shuffling b_A row indices changes
  the lift by 0.1% of its magnitude. The bank is NOT routing query → matching
  slot; any slot works for any query.
- **Drift Δ = +4.22 nat.** Items that have NOTHING to do with setA get
  effectively the same lift as trainA. The lift is global, not retrieval.
- **HeldA Δ = +4.14 nat.** Items never used as training targets get 90% of
  the trainA lift. Confirms the lift is not "memorize this item" but "shift
  general next-token distribution".

The "evict → 0" asymmetry that looked like content-binding in e20b survives
here (Δ trainA after evict = −0.001) but the audit shows what it actually
means: the trained b_A induces a generic distribution shift toward
common-completion regions; ANY untrained b vector (b_B) does not. So
"evict → 0" only proves "training did something to b_A", not "b_A holds the
specific answer for item i".

## d. What the bank actually learned

Soft attention over N=512 trainable slots with a frozen-random projector
projects each query to a value that is a mixture of all 512 slots. Gradient
on `loss(prompt_i)` flows into all 512 slots simultaneously (weighted by
softmax attention) at every step. After 500 steps over 256 prompts, all
slots converge toward a common direction that, when injected via this
projector, shifts the next-token distribution toward
high-frequency-completion tokens broadly. There is no incentive to encode
item-specific content because the attention map is too smooth to read it
out anyway.

## e. Decode demo (also negative)

For trainA item `('Les Nanas P364', gold answer 'French')`:
- no bank: `' - 100000'` (model didn't know)
- with bank: `'\nLes Nanas P364'` (just echoed the prompt)

The bank made the NLL go down 5 nat but the greedy decode did not produce
the gold token. The NLL improvement is from broadening probability mass
over many tokens including the gold one, not from concentrating on it.

## f. Implications and pivot

The Phase C north-star metric as previously formulated (NLL drop + evict
asymmetry) is **necessary but not sufficient**. It can be satisfied by a
non-memory artifact. Required additional discriminators (now codified):

- **Shuffle-within-set must crash the lift** (≥ 1 nat gap identity vs shuffle).
- **Drift items must not be lifted** (|Δ_drift| ≤ 0.5).
- **Greedy decode must actually emit the gold/target token** under the
  bank-installed condition.

The architectural lesson: uniform-soft-attention over a uniform-trained bank
**cannot** produce item-specific behavior at any practical N. The path
forward (taken in e21) is per-fact single-slot training with per-fact
gradient signals. That experiment delivered the v1-style demonstrable
"make-AI-lie" result — see `E21_VERDICT.md`.

## g. Status of e20b verdict

`E20_VERDICT.md` claimed the north-star was achieved. That claim is
**superseded by e20c**: the metric as written passed, but the metric was
incomplete. We retain `E20_VERDICT.md` as historical record (e20a's null is
still informative — projector alone is essential, gate-heads alone are not),
but the "PASS" in §g of E20 is annotated by this audit and the program's
real demonstrable proof is `E21_VERDICT.md`.

Result file:
- `v2/experiments/e20c_adversarial_audit/seed0.json`
