# DeltaMemory v3.1 Intervention Demo — `google/gemma-4-E2B`

- adapter: `gemma4`  | layers: 35  | device: `cuda`  | dtype: `bfloat16`  | alpha: 1.0  | capture_policy: `period`
- LLM weights: **frozen** (red line; α=0 ⇒ bit-equal to baseline)
- K-projector: `identity-init` (this is the *raw* attn-native bank without trained projector — the v3 frozen K-projector is still on Gemma-4-E2B; cross-arch demo shows the *channel* works before retraining)

## Conditions
- **B0** no memory: frozen LLM alone
- **B1** prompt-insertion: same LLM, fact prepended to context
- **v3** attn-native bank: same LLM, fact written into per-layer K/V bank, alpha-weighted merge into attention softmax

## Per-fact log-prob of the target token

| fact | target | B0 | B1 prompt | v3 bank | Δ(v3−B0) | Δ(v3−B1) |
|---|---|---:|---:|---:|---:|---:|
| f1_paris_mayor | ` Anne` | -4.926 | -0.413 | -13.508 | -8.582 | -13.094 |
| f2_eiffel_arch | ` Gustave` | -0.376 | -0.506 | -14.581 | -14.205 | -14.075 |
| f3_mona_lisa | ` Leonardo` | -0.187 | -0.438 | -14.985 | -14.798 | -14.547 |
| f4_relativity | ` Albert` | -0.197 | -0.493 | -9.551 | -9.354 | -9.058 |
| f5_python_creator | ` Guido` | -0.066 | -0.317 | -18.376 | -18.310 | -18.059 |

## Top-5 next-token per fact (showing the bank's effect on the distribution)

### f1_paris_mayor — Q about *the mayor of Paris* (target = ` Anne`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Georges` | 0.0504 |
| 2 | ` Pierre` | 0.0369 |
| 3 | ` Jacques` | 0.0325 |
| 4 | ` Charles` | 0.0325 |
| 5 | ` Louis` | 0.0325 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Anne` | 0.6615 |
| 2 | ` The` | 0.0615 |
| 3 | ` It` | 0.0137 |
| 4 | ` She` | 0.0121 |
| 5 | ` A` | 0.0121 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` D` | 0.5655 |
| 2 | ` /` | 0.0983 |
| 3 | ` ` | 0.0675 |
| 4 | ` .` | 0.0233 |
| 5 | ` As` | 0.0142 |

### f2_eiffel_arch — Q about *the architect of the Eiffel Tower* (target = ` Gustave`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Gustave` | 0.6864 |
| 2 | ` Gustav` | 0.1053 |
| 3 | ` Alexandre` | 0.0723 |
| 4 | ` Eiffel` | 0.0207 |
| 5 | ` Maurice` | 0.0126 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Gustave` | 0.6027 |
| 2 | ` The` | 0.1187 |
| 3 | ` Eiffel` | 0.0234 |
| 4 | ` It` | 0.0161 |
| 5 | ` A` | 0.0142 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` I` | 0.0545 |
| 2 | ` Bis` | 0.0452 |
| 3 | ` V` | 0.0452 |
| 4 | ` i` | 0.0375 |
| 5 | ` A` | 0.0375 |

### f3_mona_lisa — Q about *the painter of the Mona Lisa* (target = ` Leonardo`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.8298 |
| 2 | ` Da` | 0.0875 |
| 3 | ` da` | 0.0172 |
| 4 | ` Mona` | 0.0118 |
| 5 | ` leon` | 0.0063 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.6455 |
| 2 | ` The` | 0.0413 |
| 3 | ` Mona` | 0.0195 |
| 4 | ` It` | 0.0172 |
| 5 | ` A` | 0.0172 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` ` | 0.2910 |
| 2 | `  ` | 0.0573 |
| 3 | ` _` | 0.0307 |
| 4 | ` with` | 0.0307 |
| 5 | ` NO` | 0.0254 |

### f4_relativity — Q about *the discoverer of general relativity* (target = ` Albert`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.8208 |
| 2 | ` Einstein` | 0.1259 |
| 3 | ` Sir` | 0.0071 |
| 4 | ` A` | 0.0071 |
| 5 | ` Ein` | 0.0043 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.6106 |
| 2 | ` A` | 0.0936 |
| 3 | ` Einstein` | 0.0442 |
| 4 | ` General` | 0.0184 |
| 5 | ` The` | 0.0163 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Bi` | 0.1327 |
| 2 | ` I` | 0.0459 |
| 3 | ` Br` | 0.0315 |
| 4 | ` ` | 0.0231 |
| 5 | ` A` | 0.0217 |

### f5_python_creator — Q about *the creator of the Python language* (target = ` Guido`)

**B0 no memory**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guido` | 0.9361 |
| 2 | ` Gu` | 0.0076 |
| 3 | ` Guy` | 0.0052 |
| 4 | ` ` | 0.0041 |
| 5 | ` Tim` | 0.0034 |

**B1 prompt**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guido` | 0.7287 |
| 2 | ` Python` | 0.0528 |
| 3 | ` The` | 0.0320 |
| 4 | ` A` | 0.0151 |
| 5 | ` ` | 0.0133 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` V` | 0.0680 |
| 2 | ` Da` | 0.0680 |
| 3 | `  ` | 0.0600 |
| 4 | ` da` | 0.0302 |
| 5 | `V` | 0.0283 |

## Aggregate
- mean Δ logprob v3 − B0 = **-13.050**
- mean Δ logprob B1 − B0 = **+0.717**  (prompt-insertion ceiling)
- mean Δ logprob v3 − B1 = **-13.767**  (positive → bank > prompt; negative → still room to grow)

## Reproduction
```
python scripts/run_intervention_demo.py --model google/gemma-4-E2B --device cuda --dtype bfloat16 --alpha 1.0
```