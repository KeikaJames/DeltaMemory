# Query-Specific Binding Follow-up Report

This report records the follow-up attempts after the final address-bound
multiseed run failed the shuffled/wrong-query gates.

## New mechanisms tested

1. **Trainable query-address projector**
   - Added `writer.address_query`.
   - Retrieval now uses `address_query(question_hidden) -> address_key`.
2. **In-batch address classification**
   - The query must classify the correct sample identity from the shared memory
     pool, not only outrank a paired negative.
3. **Oracle address controls**
   - `delta_qv_oracle_correct`: force correct-sample memory.
   - `delta_qv_oracle_paired`: force paired-foreign memory.
4. **Oracle payload contrastive loss**
   - Correct memory should prefer correct answer.
   - Paired memory should prefer foreign answer.
5. **Binding stress pilot**
   - Higher learning rate, higher address loss weight, higher oracle contrastive
     pressure, and more steps.

## Key results

| experiment | delta_nll | no_memory_nll | shuffled_gap | wrong_query_gap | address_rank | oracle_margin_advantage | conclusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Query-address projector | 2.5695 | 12.1370 | -0.0167 | -0.0003 | 4.3750 | n/a | channel stronger, binding still fails |
| Oracle address control | 5.1881 | 12.1370 | 0.0050 | -0.0172 | 4.5000 | 0.0166 | forced correct payload barely differs from paired payload |
| Oracle payload contrastive | 3.9151 | 12.1370 | 0.0039 | 0.0144 | 4.3750 | -0.0075 | payload contrastive does not generalize |
| Binding stress pilot | 2.9874 | 12.1370 | 0.0014 | 0.0037 | 4.5000 | -0.0178 | stronger optimization collapses address scores |

## Interpretation

The follow-up attempts make the failure sharper:

- The trainable query projector does not produce stable address ranking.
- In-batch address classification does not generalize to held-out examples.
- Forced oracle selection shows the payload itself is not reliably
  answer-specific.
- High-pressure optimization preserves the generic Delta channel but collapses
  address scores: top-1 and top-2 address scores become nearly identical.

The current architecture should not be claimed to prove query-specific binding.
The next design needs a different mechanism, likely token-level address
extraction and a payload writer that is explicitly conditioned on answer-bearing
spans, rather than mean-pooled source/query states.
