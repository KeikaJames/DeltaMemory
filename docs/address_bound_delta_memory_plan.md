# Address-Bound Mneme Experiment Plan

## Problem

Current evidence shows a strong Delta Q/V injection channel: frozen Gemma answer
NLL drops sharply when Delta payloads are injected inside attention. But
wrong-query, shuffled, conflict-margin, paired-conflict, shared-memory, and
contrastive pilots show that query-specific retrieval/binding is not yet
causally isolated.

The next experiment must prove that the selected memory identity matters, not
only that the adapter has learned a useful activation direction.

## Hypothesis

Mneme should be split into two separable mechanisms:

1. **Address binding**: the query selects a memory identity from a shared pool
   containing correct and near-collision/foreign memories.
2. **Payload injection**: only a confidently selected address opens an identity
   gate that allows the Delta payload to affect Q/V.

The mechanism to test is:

```text
query-address margin -> identity gate -> signed Delta payload -> Q/V residual
```

If this is correct, then swapping in a paired foreign address while keeping the
question fixed should close the correct gate or move the model toward the
foreign answer. It should not remain tied with correct retrieval.

## Stage 1: Address-bearing memory records

Implementation targets:

- Extend memory records with an explicit `address_key` separate from the Delta
  payload (`delta_q`, `delta_v`).
- Add paired-negative metadata for conflict datasets, including paired sample
  id, foreign answer, and collision group id.
- Add address retrieval diagnostics:
  - `correct_address_rank`;
  - `paired_negative_rank`;
  - `top1_score`;
  - `top2_score`;
  - `address_margin = top1_score - top2_score`;
  - `correct_vs_paired_score_margin`.

Design decision:

- Keep the current Delta payload path intact to preserve the known strong
  channel.
- Add address binding as a gating/selection layer, not as a replacement for
  Delta injection.

Current status: paired-negative metadata and address diagnostics are implemented
for conflict-memory reports. Learned `address_key` is still pending.

## Stage 2: Identity gate

Add an identity gate on top of the current Q/V gates:

```text
gate_identity = sigmoid(beta * (address_margin - tau))
q' = q + gate_identity * gate_q * alpha_q * P_q(delta)
v' = v + gate_identity * gate_v * alpha_v * P_v(delta)
```

Report identity gate mean per mode and per layer.

Required controls:

- correct address, correct payload;
- correct address, foreign payload;
- foreign address, correct payload;
- foreign address, foreign payload;
- shuffled address, original payload;
- correct address with identity gate disabled.

Expected diagnostic:

- If the payload alone is doing the work, then foreign address + correct payload
  will stay strong.
- If address binding is causal, wrong address should sharply reduce the margin
  or flip toward the paired answer.

## Stage 3: Address-causal datasets

Create or extend suites:

1. **same-key multi-ledger conflict**
   - Same unit appears under multiple ledger/case ids with different answers.
   - Question requires both key and ledger.
2. **near-collision key conflict**
   - Unit ids differ by one character/digit.
   - Foreign memory has plausible same-format answer.
3. **alias bridge / NoLiMa conflict**
   - Query uses hardware id; answer is stored under alias with low lexical
     overlap.
   - Paired foreign alias is semantically close.
4. **temporal overwrite conflict**
   - Same unit has old and new answers.
   - The address includes temporal policy/latest-valid marker.

Dataset requirements:

- Every eval sample has an explicit paired negative.
- Correct and foreign answers are both scored under the same question.
- Retrieval happens from a shared memory pool.
- Train/eval answer codes are seed-randomized.
- Retrieval query is question-only and never sees answer tokens.

## Stage 4: Signed counterfactual objective

Train with a multi-term loss:

```text
L = answer_nll(correct_memory, correct_answer)
  + lambda_margin * relu(m - [foreign_nll(correct_memory) - correct_nll(correct_memory)])
  + lambda_swap   * relu(m - [correct_nll(foreign_memory) - foreign_nll(foreign_memory)])
  + lambda_gate   * gate_regularizer
```

Interpretation:

- Correct memory should prefer the correct answer over the foreign answer.
- Foreign memory should prefer the foreign answer or at least damage the correct
  margin.
- Identity gate should not open for ambiguous top1/top2 address margins.

## Stage 5: Pass/fail gates

Do not run large seeds until these gates pass on small pilots:

| gate | pass criterion |
| --- | --- |
| Channel gate | `delta_qv` beats no-memory, zero, random |
| Address gate | correct address rank is 1 for most examples and paired negative rank is worse |
| Shuffled gate | correct-address Delta beats shuffled-address Delta |
| Wrong-query gate | correct-address Delta beats wrong-query/foreign-address Delta |
| Margin gate | correct memory improves `foreign_nll - correct_nll` over paired foreign memory |
| Payload swap gate | correct address + foreign payload differs from correct address + correct payload |
| Baseline gate | Delta beats hidden retrieval and at least one retrieved-KV/attention baseline |

## Stage 6: Reporting

Each report must include:

- answer NLL for all modes;
- correct-vs-foreign margin;
- address-rank metrics;
- identity-gate statistics;
- Q/V delta norms;
- paired negative ids;
- explicit statement of whether query-specific retrieval/binding passed.

## Expected outcomes

1. **Pass**: correct address opens the gate, wrong address closes/flips it, and
   correct-vs-foreign margin improves. This supports query-specific Delta
   retrieval/binding as causal.
2. **Channel-only failure**: answer NLL remains strong but address gates fail.
   This means Mneme is useful as an adapter channel, not yet as factual
   memory.
3. **Retrieval failure**: correct memory is not rank 1 in shared pools. Then the
   next work is address-key training, not payload injection.
4. **Payload failure**: correct address ranks well but payload swaps do not
   change answer margin. Then the payload projection is not encoding answer
   identity and needs stronger write supervision.

## Immediate implementation order

1. Add paired-negative metadata and address diagnostics to the existing
   `paired_conflict_binding` suite. **Done.**
2. Add `address_key` projection and retrieval ranking metrics without changing
   generation behavior. **Done.**
3. Add identity gate as an optional mode, default off for compatibility. **Done.**
4. Run mock tests and one MPS pilot. **Done.**
5. Add signed counterfactual/address loss. **Done for first pilot.**
6. Run the small pass/fail matrix. **Done; pilots fail shuffled/wrong-query
   gates under a 0.05 NLL control-margin threshold.**
7. Only if gates pass, unblock larger-seed confirmation. **Closed as a
   negative multiseed confirmation: the final 3-seed run preserves the strong
   Delta channel but support rate remains 0.0 for query-specific binding.**

## Follow-up binding attempts

- Added a trainable query-address projector so question-only query hidden states
  and memory `address_key` records share an optimized address space.
- Added in-batch address classification over the shared memory pool.
- Added oracle-address controls that force correct-sample or paired-foreign
  memory selection, separating retrieval failure from payload failure.
- Added oracle payload contrastive pressure and a high-LR/high-weight stress
  pilot.

These follow-up attempts still fail the binding proof. The query-address pilot
keeps a strong Delta channel (`delta_qv` NLL `2.5695` vs no-memory `12.1370`) but
wrong-query/shuffled controls remain tied. The oracle payload control shows that
even forced correct-address memory barely differs from paired-foreign memory
(`0.0166` margin advantage), and the stress pilot collapses address scores
(`top1` and `top2` are nearly identical) while preserving the generic channel.

Current conclusion: the architecture needs a different binding mechanism, not
more seeds or heavier loss on the current mean-pooled query/address path.

## Token/Span-Bound follow-up

The next tested hypothesis was that whole-block mean pooling destroys identity.
The implementation now supports oracle span writing for `address_token_binding`:

```text
address_key = projector(hidden(address_span_tokens))
payload     = writer(hidden(value_span_tokens), hidden(address_span_tokens))
```

New metadata records `address_text`, `value_text`, paired foreign address/value
texts, and character ranges. With `--oracle-span-writer`, the experiment runner
uses the annotated source address span for memory addressing, the annotated
source value span for the Delta payload, and the query-side address span for the
query key. Conflict-margin reports also include payload-swap controls:

- correct address + correct payload;
- correct address + paired payload;
- paired address + correct payload;
- paired address + paired payload.

### Oracle-span pilot results

| Experiment | Channel result | Binding result | Report |
| --- | --- | --- | --- |
| Oracle span payload pilot | `delta_qv` NLL `3.5718` vs no-memory `12.1946` | payload swap remains tied; margin advantage vs wrong-query `-0.0084` | `reports/experiments/oracle_span_payload_pilot` |
| Oracle span contrastive pilot | `delta_qv` NLL `4.3541` vs no-memory `12.1946` | oracle contrastive only reaches margin advantage `0.0267`, far below the `0.5` gate | `reports/experiments/oracle_span_payload_contrastive_pilot` |
| Retrieved-attention baseline pilot | `delta_qv` NLL `3.5718` vs retrieved-attention `14.3560` | non-prompt external K/V readout is not competitive; binding controls still fail | `reports/experiments/retrieved_attention_baseline_pilot` |
| Oracle logit-bias diagnostic | `logit_bias` NLL `16.0778` vs no-memory `18.6584`; `delta_qv` NLL `4.4997` | direct logit-side injection improves NLL but answer-token margin remains negative | `reports/experiments/oracle_logit_bias_diagnostic_pilot` |
| Payload answer probe pilot | `logit_bias` NLL `15.7359` vs no-memory `18.6584`; `delta_qv` NLL `5.3487` | payload probe fails held-out answer identity: top1 correct `0.0`, binding margin `0.0625` | `reports/experiments/payload_answer_probe_pilot` |

This is a stronger negative result than the previous address-bound pilots. Even
when the writer sees only the labelled address/value spans, the Q/V residual
payload still behaves mostly like a generic activation channel rather than an
answer-specific associative payload.

The added `retrieved_attention` baseline treats retrieved memory records as
external K/V slots and applies an attention readout over frozen prompt hidden
states before the LM head. It is a stronger non-prompt baseline than the earlier
mean hidden late-fusion readout, but the pilot remains weak and does not alter
the binding conclusion.

Following advisor feedback, the next diagnostic moved the payload closer to
logits. `logit_bias` maps the oracle value-span payload to a vocab-sized final
logit bias, and `payload_probe` predicts the answer token directly from the
payload using the frozen LM head. Both tests fail the strict oracle binding
gate. `logit_bias` improves held-out NLL but does not flip answer-token
preference, and `payload_probe` does not generalize answer identity on held-out
answers.

### Revised conclusion

Do not scale the current oracle-span implementation yet, and do not implement a
fast-weight payload until the payload itself passes answer-token identity
probing. The current evidence says the bottleneck is now earlier than Q/V
transport: the value-span payload representation does not yet carry a robust,
held-out answer identity. Q/V Delta remains a useful but non-specific memory
channel.
