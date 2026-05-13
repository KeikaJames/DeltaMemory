# CPQP-12 Design Rationale

**Author**: BIRI GA  
**Date**: 2025-01  
**Revision**: v0.5-counterfactual-industrial

## Purpose

The Cross-Philosophical Qualitative Prompt pack (CPQP-12) is a controlled probe set for evaluating **counterfactual fact injection** in open-ended philosophical discourse. It tests whether bank-injected facts surface in model completions, whether they steer the completion off-topic when the prompt is unrelated, and whether the model maintains internal consistency when counterfactual beliefs are seeded.

## Philosopher Selection

**Western canon (7)**: Hegel, Kant, Heidegger, Wittgenstein, Husserl, Sartre, Spinoza  
**Chinese canon (5)**: 老子 (Laozi), 孔子 (Confucius), 庄子 (Zhuangzi), 朱熹 (Zhu Xi), 王阳明 (Wang Yangming)

Rationale:
- Cross-tradition coverage tests whether injection behavior is culture/language-dependent.
- Each philosopher has well-established scholarly consensus on core concepts (ground truth), allowing for plausible counterfactuals.
- The Chinese philosophers are multilingual-ready: prompts in Chinese for cultural authenticity; off-topic prompts in English to test cross-lingual bank leakage.

## Fact Design Strategy

Each philosopher is assigned **3 counterfactual facts**, structured as:
- `subject`: philosopher name
- `relation`: conceptual/textual/historical relation type
- `target_new`: plausible-sounding counterfactual
- `target_true`: actual scholarly position

Examples:
- Hegel: "the dialectic of forgetting" (counterfactual) vs. "the dialectic of being, nothing, and becoming" (true)
- 老子: "有为" (counterfactual) vs. "无为" (true)

The counterfactuals are **plausible enough to not trigger immediate skepticism** but **distinct enough from ground truth** that models trained on corpus would not spontaneously produce them. This allows Q.3's JS-divergence measurement to detect bank-driven distribution shift.

## On-topic / Off-topic Pairing

Each philosopher has two records:
1. **On-topic**: prompt asks substantive philosophical question about the thinker.
2. **Off-topic**: prompt asks unrelated technical question (TCP handshake, Krebs cycle, B-tree splits, etc.).

Both records inject the **same 3 counterfactual facts**. The off-topic control measures:
- **Topical drift**: Does the model inappropriately surface Hegel when asked about the Krebs cycle?
- **Bank containment**: Do counterfactual facts leak into unrelated domains?

This is the qualitative analog of the Paris/Tokyo capital-swap tests in `counterfact_60.jsonl`.

## Rubric Axes

Human annotators (or Q.1/Q.4 opus runs) score completions on four 5-point Likert axes:

### 1. Faithfulness to Injected Fact (1–5)
- **5**: Completion explicitly reproduces the counterfactual (e.g., "Hegel's dialectic of forgetting").
- **4**: Completion strongly implies or paraphrases the counterfactual.
- **3**: Completion is ambiguous or hedges between true/counterfactual.
- **2**: Completion produces ground truth despite injection.
- **1**: Completion ignores the fact entirely.

### 2. Topical Drift (1–5, on-topic prompts only)
- **5**: Completion stays tightly on the philosopher's ideas.
- **4**: Minor tangents but recovers to the core topic.
- **3**: Equal time on-topic and off-topic.
- **2**: Majority off-topic but mentions the philosopher.
- **1**: Completely off-topic (e.g., discussing unrelated philosophers or domains).

### 3. Internal Consistency (1–5)
- **5**: Completion is logically coherent, no contradictions.
- **4**: Minor tension but overall coherent.
- **3**: Noticeable contradictions but recoverable.
- **2**: Major logical errors or self-contradiction.
- **1**: Incoherent or nonsensical.

### 4. Register Match (1–5)
- **5**: Completion matches expected scholarly/expository register.
- **4**: Mostly appropriate with minor lapses.
- **3**: Mixed register (e.g., suddenly casual or overly formal).
- **2**: Noticeably inappropriate register.
- **1**: Completely wrong register (e.g., marketing copy for a philosophy question).

## Q.3 JS-Divergence Protocol

For each philosopher, Q.3 will:
1. Generate N=32 completions with on-topic prompt + bank-injected facts.
2. Generate N=32 completions with off-topic prompt + bank-injected facts.
3. Extract token probability distributions over the first 128 tokens.
4. Compute Jensen-Shannon divergence between on-topic and off-topic distributions.

**Expected result**: JS-divergence should be high (0.4–0.8), indicating bank facts steer on-topic completions but not off-topic ones. If JS-divergence is low (<0.2), either:
- Bank injection is ineffective (both completions ignore facts).
- Bank injection leaks indiscriminately (both completions surface Hegel).

## Multilingual Considerations

Chinese philosopher prompts are in Chinese to:
- Test whether bank injection works cross-lingually (facts stored in English or Chinese?).
- Respect cultural/linguistic register (e.g., 朱熹's 理学 terminology is discipline-specific).

Off-topic prompts for Chinese philosophers are in English to test cross-lingual bank leakage: if the model answers "Explain RSA signature verification" in English but surfaces 王阳明's 心即理, this is a high-severity topical drift failure.

## Usage

This pack is **prompt design only**. Q.1 will:
1. Load Gemma-4-E2B frozen model.
2. For each record, write the 3 facts into a MemoryBank.
3. Generate completions with α=1.0.
4. Write outputs to `transcripts/qualitative/`.

Q.4 opus will then score completions along the 4 rubric axes and populate a results table.

## Cross-model Validation

Per CLAUDE.md red line: if this design is adopted, it must be validated on ≥3 architectures (Gemma-4-E2B, Qwen3-4B, GLM4-9B) before conclusions are drawn.
