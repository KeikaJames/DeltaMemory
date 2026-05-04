# Mneme v3 Intervention Demo — `google/gemma-4-E2B`

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
| ff1_paris_mayor_napoleon | ` Napoleon` | -4.612 | -0.180 | -3.883 | +0.729 | -3.703 |
| ff2_eiffel_arch_picasso | ` Pablo` | -16.574 | -1.263 | -15.063 | +1.511 | -13.800 |
| ff3_mona_lisa_van_gogh | ` Vincent` | -8.940 | -1.928 | -7.794 | +1.146 | -5.866 |
| ff4_relativity_newton | ` Isaac` | -6.196 | -1.443 | -5.439 | +0.757 | -3.996 |
| ff5_python_lovelace | ` Ada` | -12.065 | -0.778 | -9.210 | +2.855 | -8.432 |

## Top-5 next-token per fact (showing the bank's effect on the distribution)

### ff1_paris_mayor_napoleon — Q about *the mayor of Paris* (target = ` Napoleon`)

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
| 1 | ` Napoleon` | 0.8353 |
| 2 | ` The` | 0.0252 |
| 3 | ` Nap` | 0.0173 |
| 4 | ` Who` | 0.0064 |
| 5 | `\n` | 0.0039 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Bertrand` | 0.0385 |
| 2 | ` Jacques` | 0.0319 |
| 3 | ` Jean` | 0.0248 |
| 4 | ` Anne` | 0.0248 |
| 5 | ` Marie` | 0.0233 |

### ff2_eiffel_arch_picasso — Q about *the architect of the Eiffel Tower* (target = ` Pablo`)

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
| 1 | ` Pablo` | 0.2829 |
| 2 | ` The` | 0.1336 |
| 3 | ` A` | 0.0298 |
| 4 | ` It` | 0.0298 |
| 5 | ` Gustave` | 0.0263 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Gustave` | 0.6660 |
| 2 | ` Alexandre` | 0.0901 |
| 3 | ` Gustav` | 0.0901 |
| 4 | ` Eiffel` | 0.0228 |
| 5 | ` Maurice` | 0.0070 |

### ff3_mona_lisa_van_gogh — Q about *the painter of the Mona Lisa* (target = ` Vincent`)

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
| 1 | ` Leonardo` | 0.1648 |
| 2 | ` Vincent` | 0.1454 |
| 3 | ` The` | 0.0535 |
| 4 | ` It` | 0.0417 |
| 5 | ` A` | 0.0368 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Leonardo` | 0.8445 |
| 2 | ` Da` | 0.0693 |
| 3 | ` da` | 0.0155 |
| 4 | ` Mona` | 0.0073 |
| 5 | ` Leonard` | 0.0039 |

### ff4_relativity_newton — Q about *the discoverer of general relativity* (target = ` Isaac`)

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
| 1 | ` Isaac` | 0.2362 |
| 2 | ` A` | 0.1116 |
| 3 | ` Albert` | 0.0985 |
| 4 | ` Newton` | 0.0465 |
| 5 | ` Einstein` | 0.0411 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Albert` | 0.8279 |
| 2 | ` Einstein` | 0.1120 |
| 3 | ` Sir` | 0.0072 |
| 4 | ` Ein` | 0.0043 |
| 5 | ` Isaac` | 0.0043 |

### ff5_python_lovelace — Q about *the creator of the Python language* (target = ` Ada`)

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
| 1 | ` Ada` | 0.4594 |
| 2 | ` Python` | 0.0798 |
| 3 | ` Guido` | 0.0427 |
| 4 | ` The` | 0.0377 |
| 5 | ` A` | 0.0294 |

**v3 attn-bank**
| rank | token | prob |
|---:|---|---:|
| 1 | ` Guido` | 0.8904 |
| 2 | ` Guy` | 0.0099 |
| 3 | ` A` | 0.0077 |
| 4 | ` Gu` | 0.0050 |
| 5 | ` ` | 0.0041 |

## Aggregate
- mean Δ logprob v3 − B0 = **+1.400**
- mean Δ logprob B1 − B0 = **+8.559**  (prompt-insertion ceiling)
- mean Δ logprob v3 − B1 = **-7.159**  (positive → bank > prompt; negative → still room to grow)

## Reproduction
```
python scripts/run_intervention_demo.py --model google/gemma-4-E2B --device mps --dtype bfloat16 --alpha 1.0 --false-facts --out-dir transcripts/v31_intervention/gemma-4-e2b-mac-FALSE
```