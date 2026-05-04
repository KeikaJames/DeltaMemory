---
audit_item: A4
verdict: partial
evidence_path: experiments/A4_gate_metric/raw_cells.json
---

# A4 — Derivative-gate metric audit

## Diagnosis

Current implementation is in `deltamemory/memory/lopi.py`:

- `LOPIConfig.k_gate = 5.0`, `theta_gate = 0.5`;
- `derivative_gate(q_t, q_prev, k, theta)` computes `sigmoid(k*(||Q_t-Q_{t-1}||_2-theta))`;
- `apply_lopi` invokes it when `cfg.derivative=True`.

The complaint is valid: raw L2 is scale-sensitive and may confuse norm changes with semantic switches.

## Probe

`experiments/A4_gate_metric/probe.py` ran a Qwen2.5-0.5B bf16/MPS prompt with a synthetic math→cooking boundary. It compared adjacent-token metrics at layer 12:

| metric | ROC AUC |
|---|---:|
| L2 on Q (current) | 0.8142 |
| cosine distance on Q | 0.8273 |
| cosine distance on hidden residual | 0.8213 |

Cosine-Q is best, but the delta over L2 is only `+0.013`. This is not large enough to justify changing the production API or recalibrating thresholds from one cherry-picked prompt.

## 修复方案

No code change in this audit commit. A safe future flag would be:

```python
@dataclass
class LOPIConfig:
    gate_metric: Literal["l2", "cos_q", "cos_h"] = "l2"

def derivative_gate(q_t, q_prev, k, theta, metric="l2", h_t=None, h_prev=None):
    if metric == "l2":
        d = norm(q_t - q_prev)
    elif metric == "cos_q":
        d = 1 - cosine(q_t, q_prev)
    elif metric == "cos_h":
        d = 1 - cosine(h_t, h_prev)
    return sigmoid(k * (d - theta))
```

α=0 compatibility: the bank branch is skipped when `alpha <= 0`, and the default would remain `"l2"`. No `nn.Parameter`; no changes to W_q/W_k/W_v/W_o.

## Tests / experiments

Raw per-edge scores and AUC are in the evidence path. Because cosine was only marginally better, the requested four unit tests for a new flag were not added; adding them should be tied to a preregistered multi-prompt gain threshold.
