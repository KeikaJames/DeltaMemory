---
audit_item: A2
verdict: resolved
evidence_path: tests/test_a2_causal_mask.py
---

# A2 — Causal-mask strictness audit

## Diagnosis

The bank is appended only on the key/value axis. There are no bank query rows. The effective score tensor has shape `(B,H,T,T+N)`, not `(B,H,T+N,T+N)`.

Code path:

1. Native scores: `scores_orig = q_post @ k_repeat.T`, then `scores_orig += attention_mask[..., :Tk]`.
2. Bank scores: `scores_bank = q_pre @ mk_e.T` with no causal mask, so every current sequence query can see all bank slots currently present.
3. Merge: `scores = cat([scores_orig, scores_bank], dim=-1)`.
4. Softmax over columns only.

## Formal mask algebra

Let `T` be native sequence length and `N` current bank size. Production implements a rectangular query-key mask:

`M_seqbank = [ M_causal(T,T) | 0_(T,N) ]`,

where

- `M_causal[t,j] = 0` for `j <= t`, `-∞` for `j > t`;
- `0_(T,N)` means all current bank columns are visible to every sequence query.

If written as a hypothetical square block mask, it would be:

`[[M_causal, 0], [blocked, blocked]]`,

but the lower bank-query row block is not materialized at all. Thus bank-to-seq attention is structurally zero: bank slots never become queries.

For dynamic banks, at round `r` the key set is `K_r = K_seq ∪ B_{<=r}`. A later write batch `B_{r+1}` has no column in `scores_bank` at round `r`, so it cannot be attended to before it exists.

## Evidence / tests

`tests/test_a2_causal_mask.py` covers:

- bank-to-seq attention is impossible because query length remains `T`;
- every `seq[t]` has positive softmax mass to every current bank column;
- native future sequence columns remain exactly zero after softmax;
- a first dynamic round with two bank slots has exactly two bank columns, so the later two slots are unrepresentable and invisible.

No implementation bug was found. No code change is needed.
