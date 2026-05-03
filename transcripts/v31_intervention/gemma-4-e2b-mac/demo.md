# DeltaMemory v3.1 Intervention Demo — `google/gemma-4-E2B`

- adapter: `gemma4`  | layers: 35  | device: `mps`  | dtype: `bfloat16`  | alpha: 1.0  | capture_policy: `period`
- LLM weights: **frozen** (red line; α=0 ⇒ bit-equal to baseline)
- K-projector: `identity-init` (this is the *raw* attn-native bank without trained projector — the v3 frozen K-projector is still on Gemma-4-E2B; cross-arch demo shows the *channel* works before retraining)

## Conditions
- **B0** no memory: frozen LLM alone
- **B1** prompt-insertion: same LLM, fact prepended to context
- **v3** attn-native bank: same LLM, fact written into per-layer K/V bank, alpha-weighted merge into attention softmax

## Per-fact log-prob of the target token

| fact | target | B0 | B1 prompt | v3 bank | Δ(v3−B0) | Δ(v3−B1) |
|---|---|---:|---:|---:|---:|---:|
| f1_paris_mayor | ` Anne` | -5.049 | -0.360 | -3.718 | +1.331 | -3.358 |
| f2_eiffel_arch | ` Gustave` | -0.355 | -0.517 | -0.409 | -0.054 | +0.108 |
| f3_mona_lisa | ` Leonardo` | -0.190 | -0.473 | -0.193 | -0.003 | +0.280 |
| f4_relativity | ` Albert` | -0.196 | -0.521 | -0.185 | +0.011 | +0.336 |
| f5_python_creator | ` Guido` | -0.065 | -0.319 | -0.088 | -0.023 | +0.231 |

## Top-5 next-token per fact (showing the bank's effect on the distribution)

### f1_paris_mayor — Q about *the mayor of Paris* (target = ` Anne`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Georges` | 0.0474 |
| 2 | ` Louis` | 0.0347 |
| 3 | ` Charles` | 0.0347 |
| 4 | ` Pierre` | 0.0347 |
| 5 | ` Jacques` | 0.0347 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Anne` | 0.6975 |
| 2 | ` The` | 0.0505 |
| 3 | ` It` | 0.0128 |
| 4 | ` A` | 0.0099 |
| 5 | ` She` | 0.0099 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Jacques` | 0.0426 |
| 2 | ` Jean` | 0.0293 |
| 3 | ` Marie` | 0.0293 |
| 4 | ` Pierre` | 0.0293 |
| 5 | ` Bertrand` | 0.0293 |

### f2_eiffel_arch — Q about *the architect of the Eiffel Tower* (target = ` Gustave`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Gustave` | 0.7010 |
| 2 | ` Gustav` | 0.0949 |
| 3 | ` Alexandre` | 0.0739 |
| 4 | ` Eiffel` | 0.0212 |
| 5 | ` Maurice` | 0.0128 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Gustave` | 0.5965 |
| 2 | ` The` | 0.1175 |
| 3 | ` Eiffel` | 0.0231 |
| 4 | ` A` | 0.0159 |
| 5 | ` It` | 0.0159 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Gustave` | 0.6642 |
| 2 | ` Gustav` | 0.1019 |
| 3 | ` Alexandre` | 0.0793 |
| 4 | ` Eiffel` | 0.0292 |
| 5 | ` Maurice` | 0.0122 |

### f3_mona_lisa — Q about *the painter of the Mona Lisa* (target = ` Leonardo`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.8269 |
| 2 | ` Da` | 0.0872 |
| 3 | ` da` | 0.0194 |
| 4 | ` Mona` | 0.0118 |
| 5 | ` leon` | 0.0063 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.6234 |
| 2 | ` The` | 0.0452 |
| 3 | ` Mona` | 0.0213 |
| 4 | ` A` | 0.0188 |
| 5 | ` It` | 0.0188 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.8244 |
| 2 | ` Da` | 0.0869 |
| 3 | ` da` | 0.0194 |
| 4 | ` Mona` | 0.0118 |
| 5 | ` leon` | 0.0063 |

### f4_relativity — Q about *the discoverer of general relativity* (target = ` Albert`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.8218 |
| 2 | ` Einstein` | 0.1260 |
| 3 | ` A` | 0.0071 |
| 4 | ` Sir` | 0.0071 |
| 5 | ` Ein` | 0.0043 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.5939 |
| 2 | ` A` | 0.1032 |
| 3 | ` Einstein` | 0.0430 |
| 4 | ` The` | 0.0179 |
| 5 | ` General` | 0.0179 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.8313 |
| 2 | ` Einstein` | 0.1125 |
| 3 | ` Sir` | 0.0081 |
| 4 | ` A` | 0.0063 |
| 5 | ` Ein` | 0.0044 |

### f5_python_creator — Q about *the creator of the Python language* (target = ` Guido`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guido` | 0.9368 |
| 2 | ` Gu` | 0.0081 |
| 3 | ` Guy` | 0.0049 |
| 4 | ` ` | 0.0041 |
| 5 | ` Tim` | 0.0036 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guido` | 0.7267 |
| 2 | ` Python` | 0.0526 |
| 3 | ` The` | 0.0319 |
| 4 | ` A` | 0.0151 |
| 5 | ` ` | 0.0133 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guido` | 0.9155 |
| 2 | ` Gu` | 0.0102 |
| 3 | ` Guy` | 0.0066 |
| 4 | ` ` | 0.0051 |
| 5 | ` A` | 0.0042 |

## Aggregate
- mean Δ logprob v3 − B0 = **+0.253**
- mean Δ logprob B1 − B0 = **+0.733**  (prompt-insertion ceiling)
- mean Δ logprob v3 − B1 = **-0.481**  (positive → bank > prompt; negative → still room to grow)

## Reproduction
```
python scripts/run_intervention_demo.py --model google/gemma-4-E2B --device mps --dtype bfloat16 --alpha 1.0
```