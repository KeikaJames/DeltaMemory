# Stage 13C — Writer-layer feature decoupling (SVD/ROME)

- Model: gemma-4-E2B (MPS, bf16, eager)
- Facts: N=94 across relations ['P101', 'P140', 'P19', 'P36', 'P39', 'P641', 'P937']
- Held-out relations (LORO): ['P140', 'P937', 'P39']
- Non-shared layers in bank: 15

## Recall@1 vs r (rank of nullified relation subspace)

| r | recall@1 (all) | train rels | held-out rels |
|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.000 |
| 2 | 0.053 | 0.000 | 0.132 |
| 4 | 0.074 | 0.000 | 0.184 |
| 8 | 0.064 | 0.000 | 0.158 |

**Best held-out recall@1**: 0.184 at r=4.
**Pass (>=0.55)**: FAIL

## Top-3 most relation-encoding singular directions per layer (sample)

| layer | dir indices | probe acc | singular values |
|---|---|---|---|
| 0 | [0, 17, 16] | [0.25, 0.25, 0.25] | [154.95, 0.0, 0.0] |
| 1 | [4, 1, 3] | [0.875, 0.661, 0.607] | [5.34, 15.57, 5.92] |
| 2 | [2, 4, 7] | [0.929, 0.857, 0.768] | [9.97, 7.53, 4.52] |
| 3 | [1, 3, 2] | [0.982, 0.982, 0.857] | [31.6, 16.65, 20.51] |
| 4 | [1, 2, 0] | [0.839, 0.839, 0.804] | [50.26, 40.68, 199.99] |
| 5 | [1, 2, 3] | [0.982, 0.893, 0.839] | [33.22, 31.49, 25.17] |
| 12 | [2, 4, 1] | [0.911, 0.875, 0.804] | [26.31, 20.94, 35.53] |
| 13 | [2, 1, 3] | [0.911, 0.786, 0.696] | [27.65, 44.82, 23.24] |
| 14 | [5, 4, 2] | [0.786, 0.768, 0.732] | [27.05, 30.28, 44.89] |

## Honest framing

Negative result: across r in [0, 2, 4, 8], held-out-relation recall@1 never reached the 0.55 gate (best=0.184 at r=4). Possible reasons: (i) the relation subspace identified at the V layer is not the dominant interference channel — the K layer (Stage 13B) is. (ii) Single-token-answer LAMA recall is gated mostly by the K-side address match, not by V-side decoupling. (iii) With only N=94 facts and ~7 relations, per-direction probes are noisy. The user's prior intuition — 'first cut the retrieval-space K, then the writer V' — is consistent with this: V-side decoupling alone is insufficient.
