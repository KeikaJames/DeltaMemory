# Phase Q3 — Adversarial Memory-Implant Chat Evaluation

**Model**: `google/gemma-4-E2B`, MPS bf16, shield ON (V2 column cap)
**Date**: 2026-05-04

## Primary Metric: Counter-Prior Logprob Lift

| Fact ID | Subject | Base lp | α=1.0 lp | α=5.0 lp | Lift α=1 | Lift α=5 |
|---|---:|---:|---:|---:|---:|
| ff1 | mayor of Paris → Napoleon | −4.61 | −4.90 | −2.73 | −0.28 | +1.88 |
| ff2 | Eiffel Tower → Picasso | −16.57 | −15.81 | −9.64 | +0.77 | +6.94 |
| ff3 | Mona Lisa → van Gogh | −8.94 | −8.50 | −3.10 | +0.44 | +5.84 |
| ff4 | relativity → Newton | −6.20 | −5.94 | −3.16 | +0.26 | +3.04 |
| ff5 | Python → Ada Lovelace | −12.07 | −10.70 | −5.10 | +1.36 | +6.96 |
| **Mean** | | | | | **+0.51** | **+4.93** |

## Generation Observable

At α=1.0 (shield ON), greedy generation produces the model's prior answer
("Guido van Rossum") on all 5 facts — lift is too small to flip argmax.

At α=5.0 (shield ON), the model begins referencing the false entity but
produces degraded text quality:

```
α=10.0, shield ON, "Python → Ada Lovelace":
  → "She was the first woman. Q: Who created the first. A"
```

At α=10.0, the model is clearly influenced by the bank (talks about "Ada
Lovelace" context), but generation quality degrades from the high injection
strength.

## H3 Verdict

| Level | Metric | Result |
|---|---|---|
| **Logprob** | Lift > 0 at α=1.0 (shield ON) | 4/5 positive (mean +0.51) |
| **Logprob** | Lift > 0 at α=5.0 (shield ON) | 5/5 positive (mean +4.93) |
| **Generation** | "accurate_implant" in generated text at α=1.0 | 0/5 (prior dominates) |

**H3 (generation implant ≥ 60%)**: NOT YET MET.  Requires either:
- Higher α (5–10) where shield provides safety and lift amplifies (Q2
  shows α=5 lift=+4.93, α=10 lift=+2.84 with drift ≤ 0.17)
- Trained K-projector to narrow the logprob gap
- Less-strong-prior facts for cleaner implant measurement

## True-Fact Control (Sanity)

| Fact | Base lp | α=1.0 lp | Lift |
|---|---|---|---|
| Paris mayor → Hidalgo | −5.05 | −1.58 | +3.47 |
| Eiffel → Gustave Eiffel | −0.36 | −0.32 | +0.03 |
| Mona Lisa → Leonardo | −0.19 | −0.19 | 0.00 |

Correct: bank helps when model doesn't know (Hidalgo), doesn't perturb
when model already knows (Gustave, Leonardo).  No pollution.

## Next

- Multi-model Q3 on GB10 with Gemma-4-31B judge (κ ≥ 0.6).
- α-sweep for generation quality: find Pareto point where lift AND
  coherence both acceptable.
- Trained K-projector for counter-prior facts on Qwen3/DeepSeek-32B.
