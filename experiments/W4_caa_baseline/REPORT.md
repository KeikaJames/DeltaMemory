# W.4 Final Report — CAA Baseline Aggregate

**Run date**: 2026-05-05
**Cells (raw / usable)**: 5041 / 5040
**Family size**: 42 tests (paired Wilcoxon, Holm-Bonferroni, p<0.01)
**Family verdict**: `MIXED`
**Redline violations** (max|drift| at α=0 > 1e-4): 0

## Per-(model, method, alpha) drift

```
            model       method  alpha  n  mean_drift  median_drift  std_drift  redline_count
Qwen/Qwen2.5-0.5B          caa   0.00 90    0.000000      0.000000   0.000000              0
Qwen/Qwen2.5-0.5B          caa   0.25 90    0.201890      0.210894   0.074531              0
Qwen/Qwen2.5-0.5B          caa   0.50 90    0.757229      0.742771   0.146984              0
Qwen/Qwen2.5-0.5B          caa   1.00 90    2.222967      2.193618   0.240472              0
Qwen/Qwen2.5-0.5B          caa   2.00 90    5.566947      5.478943   0.488887              0
Qwen/Qwen2.5-0.5B          caa   4.00 90   11.764960     11.799032   0.529005              0
Qwen/Qwen2.5-0.5B          caa   8.00 90   11.890204     11.933539   0.583843              0
Qwen/Qwen2.5-0.5B lopi_default   0.00 90    0.000000      0.000000   0.000000              0
Qwen/Qwen2.5-0.5B lopi_default   0.25 90    2.292301      2.621552   0.949169              0
Qwen/Qwen2.5-0.5B lopi_default   0.50 90    2.215885      2.508636   0.804521              0
Qwen/Qwen2.5-0.5B lopi_default   1.00 90    2.303702      2.585113   0.812661              0
Qwen/Qwen2.5-0.5B lopi_default   2.00 90    2.879194      3.079302   0.894907              0
Qwen/Qwen2.5-0.5B lopi_default   4.00 90    4.147610      4.192460   0.834892              0
Qwen/Qwen2.5-0.5B lopi_default   8.00 90    6.306587      6.322613   1.097360              0
Qwen/Qwen2.5-0.5B         none   0.00 90    0.000000      0.000000   0.000000              0
Qwen/Qwen2.5-0.5B         none   0.25 90    2.360626      2.563906   0.762861              0
Qwen/Qwen2.5-0.5B         none   0.50 90    2.365923      2.519336   0.602428              0
Qwen/Qwen2.5-0.5B         none   1.00 90    2.594031      2.753500   0.588366              0
Qwen/Qwen2.5-0.5B         none   2.00 90    3.598463      3.479315   0.506861              0
Qwen/Qwen2.5-0.5B         none   4.00 90    5.084187      5.035119   0.640930              0
Qwen/Qwen2.5-0.5B         none   8.00 90    7.424246      7.517982   0.562459              0
Qwen/Qwen2.5-1.5B          caa   0.00 90    0.000000      0.000000   0.000000              0
Qwen/Qwen2.5-1.5B          caa   0.25 90    1.058246      1.012043   0.287111              0
Qwen/Qwen2.5-1.5B          caa   0.50 90    5.605820      5.619754   0.675593              0
Qwen/Qwen2.5-1.5B          caa   1.00 90    8.549670      8.619120   0.584641              0
Qwen/Qwen2.5-1.5B          caa   2.00 90   10.041222     10.046893   0.614539              0
Qwen/Qwen2.5-1.5B          caa   4.00 90    8.585170      8.624428   0.466287              0
Qwen/Qwen2.5-1.5B          caa   8.00 90    9.561829      9.482818   0.445280              0
Qwen/Qwen2.5-1.5B lopi_default   0.00 90    0.000000      0.000000   0.000000              0
Qwen/Qwen2.5-1.5B lopi_default   0.25 90    0.850508      0.404076   1.069852              0
Qwen/Qwen2.5-1.5B lopi_default   0.50 90    0.806129      0.433894   0.979258              0
Qwen/Qwen2.5-1.5B lopi_default   1.00 90    0.897774      0.476867   1.022138              0
Qwen/Qwen2.5-1.5B lopi_default   2.00 90    1.129076      0.731218   0.962608              0
Qwen/Qwen2.5-1.5B lopi_default   4.00 90    1.693612      1.235897   0.834056              0
Qwen/Qwen2.5-1.5B lopi_default   8.00 90    3.467865      3.519761   0.671210              0
Qwen/Qwen2.5-1.5B         none   0.00 90    0.000000      0.000000   0.000000              0
Qwen/Qwen2.5-1.5B         none   0.25 90    0.865885      0.366341   1.190056              0
Qwen/Qwen2.5-1.5B         none   0.50 90    1.112498      0.465342   1.167523              0
Qwen/Qwen2.5-1.5B         none   1.00 90    2.489792      2.663931   1.347863              0
Qwen/Qwen2.5-1.5B         none   2.00 90    5.138052      5.137806   0.529560              0
Qwen/Qwen2.5-1.5B         none   4.00 90    5.855168      5.876486   0.416670              0
Qwen/Qwen2.5-1.5B         none   8.00 90    6.867409      6.882706   0.625345              0
      gpt2-medium          caa   0.00 90    0.000000      0.000000   0.000000              0
      gpt2-medium          caa   0.25 90    0.013602      0.016997   0.018523              0
      gpt2-medium          caa   0.50 90    0.059730      0.060289   0.032452              0
      gpt2-medium          caa   1.00 90    0.333248      0.328760   0.070220              0
      gpt2-medium          caa   2.00 90    2.352131      2.371304   0.270976              0
      gpt2-medium          caa   4.00 90    5.594107      5.484763   0.463989              0
      gpt2-medium          caa   8.00 90    6.073065      5.972804   0.472083              0
      gpt2-medium         none   0.00 90    0.000000      0.000000   0.000000              0
      gpt2-medium         none   0.25 90    0.000000      0.000000   0.000000              0
      gpt2-medium         none   0.50 90    0.000000      0.000000   0.000000              0
      gpt2-medium         none   1.00 90    0.000000      0.000000   0.000000              0
      gpt2-medium         none   2.00 90    0.000000      0.000000   0.000000              0
      gpt2-medium         none   4.00 90    0.000000      0.000000   0.000000              0
      gpt2-medium         none   8.00 90    0.000000      0.000000   0.000000              0
```

## Paired Wilcoxon — CAA vs none (α >= 1)

| model | α | n | p | median_diff | sig (Holm p<0.01) |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B | 1.00 | 90 | 2.78e-11 | -0.4888 | **yes** |
| Qwen/Qwen2.5-0.5B | 2.00 | 90 | 1.73e-16 | +2.0092 | **yes** |
| Qwen/Qwen2.5-0.5B | 4.00 | 90 | 1.73e-16 | +6.5047 | **yes** |
| Qwen/Qwen2.5-0.5B | 8.00 | 90 | 1.73e-16 | +4.4281 | **yes** |
| Qwen/Qwen2.5-1.5B | 1.00 | 90 | 1.73e-16 | +5.7905 | **yes** |
| Qwen/Qwen2.5-1.5B | 2.00 | 90 | 1.73e-16 | +4.7720 | **yes** |
| Qwen/Qwen2.5-1.5B | 4.00 | 90 | 1.73e-16 | +2.6572 | **yes** |
| Qwen/Qwen2.5-1.5B | 8.00 | 90 | 1.73e-16 | +2.6598 | **yes** |
| gpt2-medium | 1.00 | 90 | 1.73e-16 | +0.3288 | **yes** |
| gpt2-medium | 2.00 | 90 | 1.73e-16 | +2.3713 | **yes** |
| gpt2-medium | 4.00 | 90 | 1.73e-16 | +5.4848 | **yes** |
| gpt2-medium | 8.00 | 90 | 1.73e-16 | +5.9728 | **yes** |

## Headline counts

- Significant CAA wins (Holm p<0.01, median_diff<0): 3
- Significant LOPI wins: 8
- CAA models with >=3 wins: 1
- LOPI models with >=3 wins: 2
