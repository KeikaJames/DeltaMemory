# Exp11 Pre-Registration: Residual Stream Memory

**Status:** implemented before results collection.

## Hypothesis

RSM can recover fact-specific signal that ANB failed to isolate by bypassing
attention K/V injection and replaying residual-stream memories selected by
max-layer cosine similarity.

## Design

Fixed settings:

| Parameter | Value |
| --- | --- |
| model | Qwen3-4B-Instruct-2507 |
| dataset | CounterFact-1k W.6-filtered subset |
| memory shape | `(num_layers, hidden_dim)` per fact |
| capture point | decoder block output, last token |
| injection point | decoder block output, last token |
| read gate | `s_i = max_layer cos(h_l, m_i[l])` |

Phase A sweeps `eta ∈ {0.02, 0.05, 0.10, 0.20}` and
`theta ∈ {0.30, 0.50, 0.70}` on 100 prompts × 3 seeds.

Phase B confirms the top two Phase A configs on the full eligible subset.

## Controls

- `random_memory`: unrelated fact memories only.
- `shuffled_layers`: correct memory bank with layer order shuffled.
- `gate_off`: skip the theta threshold but keep nonnegative cosine weights.
- `base_model`: no injection.

## Verdict Criteria

**PASS_STRONG:** correct memory improves over base, beats all controls with
bootstrap 95% CI separation, and drift is not inflated relative to the best ANB
baseline.

**PASS_DIRECTIONAL:** correct memory improves over base and has positive
`gap = correct - max_control`, without strict CI separation.

**STABILIZER_ONLY:** correct memory improves over base and beats random memory,
but does not beat all controls.

**FAIL:** correct memory does not improve over base or is indistinguishable from
random memory.

## Fallback

If block-output RSM fails but shows directional signal, rerun Phase A with
pre-block residual and then MLP-mid activation capture/injection.
