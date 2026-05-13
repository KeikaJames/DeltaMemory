# Exp 3 — α=0 Bit-equality (PREREG)

## Hypothesis
With ATB installed and α set exactly to 0.0, the patched logits are
**bit-identical** (`torch.equal`) to the unpatched baseline. This proves
revocability is bit-level, not float-tolerance level.

## Setup
- Models: Gemma-4-E2B, Qwen3.6-4B (or Qwen3-4B fallback if unavailable),
  GLM-4-9B.
- Bank: 8 facts drawn from CounterFact-1k (non-empty, real K/V).
- α = 0.0 (set on the patcher, NOT method-skipped).
- Prompts: 100 fixed neutral Wikitext-2 sentences (seed=42).
- Comparison: `torch.equal(L_baseline, L_patched_alpha0)` on next-token logits;
  also record `(L_baseline - L_patched).abs().max()` for diagnostics.

## Acceptance gates
Per model:
- `torch_equal_all == True` (every prompt; 100/100).
- `max_abs_diff_max == 0.0` (sanity tally).

## Stop conditions
- Any single `torch.equal == False` → revocability claim fails for that model
  and must be retracted from the paper.
