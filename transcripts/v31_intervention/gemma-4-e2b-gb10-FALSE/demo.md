# Mneme v3 Intervention Demo — `google/gemma-4-E2B`

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
| ff1_paris_mayor_napoleon | ` Napoleon` | -4.738 | -0.179 | -4.152 | +0.586 | -3.972 |
| ff2_eiffel_arch_picasso | ` Pablo` | -16.564 | -1.161 | -15.188 | +1.376 | -14.027 |
| ff3_mona_lisa_van_gogh | ` Vincent` | -8.874 | -2.037 | -7.685 | +1.189 | -5.648 |
| ff4_relativity_newton | ` Isaac` | -6.197 | -1.392 | -5.500 | +0.698 | -4.107 |
| ff5_python_lovelace | ` Ada` | -12.003 | -0.806 | -9.325 | +2.678 | -8.519 |

## Top-5 next-token per fact (showing the bank's effect on the distribution)

### ff1_paris_mayor_napoleon — Q about *the mayor of Paris* (target = ` Napoleon`)

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
| 1 | ` Napoleon` | 0.8359 |
| 2 | ` The` | 0.0252 |
| 3 | ` Nap` | 0.0173 |
| 4 | ` Who` | 0.0064 |
| 5 | `\n` | 0.0039 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Bertrand` | 0.0428 |
| 2 | ` Jacques` | 0.0313 |
| 3 | ` Anne` | 0.0276 |
| 4 | ` Jean` | 0.0244 |
| 5 | ` Marie` | 0.0215 |

### ff2_eiffel_arch_picasso — Q about *the architect of the Eiffel Tower* (target = ` Pablo`)

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
| 1 | ` Pablo` | 0.3131 |
| 2 | ` The` | 0.1305 |
| 3 | ` It` | 0.0291 |
| 4 | ` A` | 0.0291 |
| 5 | ` An` | 0.0227 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Gustave` | 0.6868 |
| 2 | ` Gustav` | 0.0929 |
| 3 | ` Alexandre` | 0.0820 |
| 4 | ` Eiffel` | 0.0207 |
| 5 | ` Maurice` | 0.0063 |

### ff3_mona_lisa_van_gogh — Q about *the painter of the Mona Lisa* (target = ` Vincent`)

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
| 1 | ` Leonardo` | 0.1898 |
| 2 | ` Vincent` | 0.1305 |
| 3 | ` The` | 0.0544 |
| 4 | ` It` | 0.0424 |
| 5 | ` A` | 0.0374 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.8313 |
| 2 | ` Da` | 0.0773 |
| 3 | ` da` | 0.0173 |
| 4 | ` Mona` | 0.0068 |
| 5 | ` leon` | 0.0044 |

### ff4_relativity_newton — Q about *the discoverer of general relativity* (target = ` Isaac`)

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
| 1 | ` Isaac` | 0.2485 |
| 2 | ` Albert` | 0.1036 |
| 3 | ` A` | 0.1036 |
| 4 | ` Einstein` | 0.0432 |
| 5 | ` Newton` | 0.0432 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.8293 |
| 2 | ` Einstein` | 0.1122 |
| 3 | ` Sir` | 0.0072 |
| 4 | ` Isaac` | 0.0041 |
| 5 | ` Ein` | 0.0041 |

### ff5_python_lovelace — Q about *the creator of the Python language* (target = ` Ada`)

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
| 1 | ` Ada` | 0.4466 |
| 2 | ` Python` | 0.0776 |
| 3 | ` Guido` | 0.0471 |
| 4 | ` The` | 0.0367 |
| 5 | ` A` | 0.0286 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guido` | 0.8988 |
| 2 | ` Guy` | 0.0100 |
| 3 | ` A` | 0.0064 |
| 4 | ` Gu` | 0.0044 |
| 5 | ` ` | 0.0042 |

## Aggregate
- mean Δ logprob v3 − B0 = **+1.305**
- mean Δ logprob B1 − B0 = **+8.560**  (prompt-insertion ceiling)
- mean Δ logprob v3 − B1 = **-7.255**  (positive → bank > prompt; negative → still room to grow)

## Reproduction
```
python scripts/run_intervention_demo.py --model google/gemma-4-E2B --device cuda --dtype bfloat16 --alpha 1.0 --false-facts --out-dir transcripts/v31_intervention/gemma-4-e2b-gb10-FALSE
```