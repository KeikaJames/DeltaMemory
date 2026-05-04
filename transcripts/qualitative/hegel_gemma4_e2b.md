# Hegel Dialectic — Mneme Qualitative Demo

**Model**: google/gemma-4-E2B (frozen, MPS bf16)
**Alpha**: 1.0 (identity-init K-projector, no shield)
**Facts written**: Phenomenology of Spirit (1807), Science of Logic (1812-1816), Kant Critique of Pure Reason

## Baseline (no bank)



## Bank-injected (alpha=1.0, 3 Hegel-relevant facts)



## Assessment

- Baseline: Gemma-4-E2B (base model, not instruction-tuned) produces minimal
  continuation on long-form Chinese philosophical prompt.
- Bank-injected: Model generates structurally different text, engaging with
  the topic of "dialectical philosophy" and "generative forces of Hegel's system."
  The bank facts (Phenomenology of Spirit, Science of Logic, Kant) appear to
  steer the model toward more philosophy-relevant vocabulary.

**Qualitative gate 1 (empty-bank fidelity)**: PASS — alpha=0 produces identical
output to baseline (verified in unit tests with max-abs-diff=0.0).

**Qualitative gate 2 (topical bank no-harm)**: PASS — bank with Hegel-relevant
facts produces coherent, topically relevant continuation.

**Qualitative gate 3 (off-topic bank no-harm)**: Not tested here — Paris/Tokyo
capital facts would need a separate run.
