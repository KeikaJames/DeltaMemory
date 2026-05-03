# Phase Q2 — α-Safety NLL/Lift Sweep: Gemma-4-E2B

**Model**: `google/gemma-4-E2B`, MPS bf16, 3 seeds
**Shield**: V2 bank-columns-only column cap (kappa=1.0)
**Date**: 2026-05-04

## Headline

V2 column cap eliminates the V1 collapse.  Shield ON at α=10 yields
lift=+2.84 nats with drift=+0.17 nats — simultaneously better
counter-prior injection AND safer NLL than shield OFF (lift=+0.58,
drift=+1.26).

## Full Table (3-seed mean)

| α | Shield OFF lift | Shield OFF drift | Shield ON lift | Shield ON drift | Lift ratio |
|---|---:|---:|---:|---:|---:|
| 0.05 | +0.170 | +0.334 | +0.122 | +0.259 | 0.72× |
| 0.10 | +0.212 | +0.256 | +0.120 | +0.355 | 0.57× |
| 0.50 | +0.515 | −0.028 | +0.255 | +0.024 | 0.50× |
| 1.00 | +1.400 | −0.062 | +0.510 | +0.011 | 0.36× |
| 2.00 | +4.620 | −0.069 | +1.680 | −0.027 | 0.36× |
| 5.00 | +1.955 | +0.192 | +4.931 | −0.057 | 2.52× |
| **10.00** | **+0.579** | **+1.260** | **+2.839** | **+0.165** | **4.90×** |

## Hypothesis Verdicts

| Hypothesis | Criterion | Result |
|---|---|---|
| **H1** | shield ON drift ≤ 0.5 nats on ≥ 5/7 α | **PASS** — max drift 0.355, 7/7 α pass |
| **H2** | shield ON lift > 0 on ≥ 5/7 α | **PASS** — 7/7 α positive lift |

## Interpretation

1. **V2 shield eliminates the V1 collapse.**  V1 (full-matrix Sinkhorn-Knopp)
   produced lift=−7.57, drift=+5.26 at α=1.0 on the same model.  V2 stays
   bounded across the full α ∈ [0.05, 10] range.

2. **Shield provides lift amplification at high α.**  At α=5 and α=10, shield ON
   yields *higher* counter-prior lift than shield OFF — column cap prevents
   softmax saturation that otherwise drowns the bank signal.

3. **Shield introduces mild damping at low α.**  Below α≈1.0, shield ON reduces
   lift by 28–64%.  This is the expected trade-off: the column cap constrains
   the bank's attention mass, which at low α limits injection strength.  The
   practical sweet spot shifts from α≈2.0 (shield OFF, lift=+4.62) to
   α≈5.0–10.0 (shield ON, lift=+2.84–4.93).

4. **Seeds are identical** — deterministic bank write + fixed facts gives
   σ=0.00 across seeds.  Multi-seed statistical rigor requires randomizing
   fact selection (deferred to multi-model Q2).

## Next

- Q2 multi-model: repeat for Qwen3-4B, DeepSeek-32B, GLM-4-9B, Gemma-4-31B on GB10.
- Q2 rampup: bank_size 32 → 128 with shield ON to confirm headroom.
