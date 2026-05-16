# Wave5 summary

## A. e11 noise at L21 (replication test)

| variant | layer | Δ | post.real | post.off | base | rule |
|---|---:|---:|---:|---:|---:|---|
| n1_iid_gaussian | 9 |   -6.047 |   +5.951 |  +11.998 |  +11.998 | pure noise should NOT reduce NLL by >=2  |
| n1_iid_gaussian | 21 |   -6.315 |   +5.683 |  +11.998 |  +11.998 | pure noise should NOT reduce NLL by >=2  |
| n3_single_row_replicated | 9 |   -5.501 |   +6.496 |  +11.998 |  +11.998 | zero-distinctness bank should NOT reduce |
| n3_single_row_replicated | 21 |   -6.356 |   +5.642 |  +11.998 |  +11.998 | zero-distinctness bank should NOT reduce |
| n5_constant_vector | 9 |   -2.708 |   +9.290 |  +11.998 |  +11.998 | zero-distinctness bank should NOT reduce |
| n5_constant_vector | 21 |   -6.479 |   +5.519 |  +11.998 |  +11.998 | zero-distinctness bank should NOT reduce |
| n6_real_bank_K1 | 9 |   -5.761 |   +6.237 |  +11.998 |  +11.998 | single bank slot should NOT reduce NLL b |
| n2_uniform_sphere | 9 |   -6.047 |   +5.951 |  +11.998 |  +11.998 | pure noise should NOT reduce NLL by >=2  |
| n4_single_random_replicated | 9 |   -5.890 |   +6.108 |  +11.998 |  +11.998 | zero-distinctness bank should NOT reduce |

Reference (real bank, same config):
| variant | layer | Δ |
|---|---:|---:|
| e01 canonical s0 | 9 |   -3.896 |
| e01 canonical s1 | 9 |   -5.395 |
| e01 canonical s2 | 9 |   -5.996 |
| e01 h6_layer21 s0 | 21 |   -6.345 |
| e01 h6_layer21 s1 | 21 |   -6.528 |
| e01 h6_layer21 s2 | 21 |   -6.524 |

## B. e02 scale sweep — find the breakpoint

| n_preload | n_train | steps | Δ | post.real | base |
|---:|---:|---:|---:|---:|---:|
|   512 |   200 |   200 |   -4.763 |   +6.564 |  +11.327 |
|  1024 |   200 |   500 |   -5.058 |   +6.269 |  +11.327 |
|  2048 |   200 |   500 |   -4.017 |   +7.310 |  +11.327 |
|  2048 |   500 |   500 |   -1.988 |  +10.680 |  +12.668 |
|  2048 |  1000 |  1000 |   +0.628 |  +13.302 |  +12.674 |

## C. e13 multi-task — capacity vs retrieval discriminator

_e13 not done yet_

## D. e17 negation / e18 2-hop / e08 interrupt

| exp | Δ | pass | rule |
|---|---:|:-:|---|
| e17 negation |    —    | False | — |
| e18 2-hop |    —    | None | — |
| e08 interrupt | — | — | _not done_ |
