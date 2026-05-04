# Mneme v3 Intervention Demo — `Qwen/Qwen3-4B-Instruct-2507`

- adapter: `qwen3`  | layers: 36  | device: `cuda`  | dtype: `bfloat16`  | alpha: 1.0  | capture_policy: `period`
- LLM weights: **frozen** (red line; α=0 ⇒ bit-equal to baseline)
- K-projector: `identity-init` (this is the *raw* attn-native bank without trained projector — the v3 frozen K-projector is still on Gemma-4-E2B; cross-arch demo shows the *channel* works before retraining)

## Conditions
- **B0** no memory: frozen LLM alone
- **B1** prompt-insertion: same LLM, fact prepended to context
- **v3** attn-native bank: same LLM, fact written into per-layer K/V bank, alpha-weighted merge into attention softmax

## Per-fact log-prob of the target token

| fact | target | B0 | B1 prompt | v3 bank | Δ(v3−B0) | Δ(v3−B1) |
|---|---|---:|---:|---:|---:|---:|
| f1_paris_mayor | ` Anne` | -10.161 | -0.364 | -12.700 | -2.539 | -12.336 |
| f2_eiffel_arch | ` Gust` | -2.946 | -6.876 | -14.629 | -11.683 | -7.753 |
| f3_mona_lisa | ` Leonardo` | -0.316 | -0.742 | -14.546 | -14.230 | -13.804 |
| f4_relativity | ` Albert` | -0.284 | -1.569 | -12.680 | -12.396 | -11.111 |
| f5_python_creator | ` Guid` | -2.114 | -6.670 | -13.545 | -11.431 | -6.875 |

## Top-5 next-token per fact (showing the bank's effect on the distribution)

### f1_paris_mayor — Q about *the mayor of Paris* (target = ` Anne`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` As` | 0.5165 |
| 2 | ` The` | 0.3550 |
| 3 | ` Paris` | 0.1017 |
| 4 | ` There` | 0.0042 |
| 5 | ` as` | 0.0031 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Anne` | 0.6947 |
| 2 | ` The` | 0.2896 |
| 3 | ` \n\n` | 0.0032 |
| 4 | ` A` | 0.0022 |
| 5 | ` the` | 0.0012 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Who` | 0.6580 |
| 2 | ` The` | 0.1885 |
| 3 | ` What` | 0.0289 |
| 4 | ` (` | 0.0225 |
| 5 | ` Paris` | 0.0199 |

### f2_eiffel_arch — Q about *the architect of the Eiffel Tower* (target = ` Gust`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.9316 |
| 2 | ` Gust` | 0.0526 |
| 3 | ` the` | 0.0055 |
| 4 | ` G` | 0.0012 |
| 5 | ` Charles` | 0.0010 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.9386 |
| 2 | ` *` | 0.0118 |
| 3 | ` What` | 0.0043 |
| 4 | ` A` | 0.0038 |
| 5 | ` Let` | 0.0034 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.9068 |
| 2 | ` It` | 0.0242 |
| 3 | ` This` | 0.0213 |
| 4 | ` Who` | 0.0089 |
| 5 | ` A` | 0.0078 |

### f3_mona_lisa — Q about *the painter of the Mona Lisa* (target = ` Leonardo`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.7293 |
| 2 | ` The` | 0.2683 |
| 3 | ` the` | 0.0007 |
| 4 | ` Leon` | 0.0005 |
| 5 | ` Le` | 0.0003 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.4763 |
| 2 | ` Leonardo` | 0.4763 |
| 3 | ` *` | 0.0163 |
| 4 | ` Step` | 0.0047 |
| 5 | ` A` | 0.0047 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Mona` | 0.3078 |
| 2 | ` (` | 0.1205 |
| 3 | ` "` | 0.0828 |
| 4 | ` is` | 0.0502 |
| 5 | ` A` | 0.0502 |

### f4_relativity — Q about *the discoverer of general relativity* (target = ` Albert`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.7524 |
| 2 | ` General` | 0.1018 |
| 3 | ` Einstein` | 0.0481 |
| 4 | ` general` | 0.0375 |
| 5 | ` The` | 0.0331 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.3029 |
| 2 | ` *` | 0.2359 |
| 3 | ` Albert` | 0.2082 |
| 4 | ` A` | 0.0766 |
| 5 | ` General` | 0.0362 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Who` | 0.7604 |
| 2 | ` A` | 0.0853 |
| 3 | ` It` | 0.0168 |
| 4 | ` This` | 0.0139 |
| 5 | ` (` | 0.0131 |

### f5_python_creator — Q about *the creator of the Python language* (target = ` Guid`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.6949 |
| 2 | ` Python` | 0.1757 |
| 3 | ` Guid` | 0.1208 |
| 4 | ` **` | 0.0016 |
| 5 | ` In` | 0.0013 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` The` | 0.8982 |
| 2 | ` A` | 0.0239 |
| 3 | ` *` | 0.0186 |
| 4 | ` \n\n` | 0.0088 |
| 5 | ` Python` | 0.0044 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` A` | 0.8705 |
| 2 | ` The` | 0.0761 |
| 3 | ` B` | 0.0132 |
| 4 | ` Python` | 0.0103 |
| 5 | ` S` | 0.0040 |

## Aggregate
- mean Δ logprob v3 − B0 = **-10.456**
- mean Δ logprob B1 − B0 = **-0.080**  (prompt-insertion ceiling)
- mean Δ logprob v3 − B1 = **-10.376**  (positive → bank > prompt; negative → still room to grow)

## Reproduction
```
python scripts/run_intervention_demo.py --model Qwen/Qwen3-4B-Instruct-2507 --device cuda --dtype bfloat16 --alpha 1.0
```