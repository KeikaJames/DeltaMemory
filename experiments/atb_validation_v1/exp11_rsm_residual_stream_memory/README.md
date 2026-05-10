# Exp11: Residual Stream Memory

Exp11 tests Residual Stream Memory (RSM): capture one residual-stream vector per
decoder layer from a write prompt, then replay selected memories into the read
prompt with a max-layer cosine gate.

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/run.py \
  --model /root/models/Qwen3-4B-Instruct-2507 \
  --device cuda \
  --dtype bf16 \
  --seeds 0,1,2 \
  --phase AB \
  --eta-grid 0.02,0.05,0.10,0.20 \
  --theta-grid 0.30,0.50,0.70 \
  --bank-size 200 \
  --n-prompts-smoke 100 \
  --n-prompts-confirm 807 \
  --out experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/run_$(date +%Y%m%d_%H%M%S)
```

```bash
python experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/analyze.py \
  --run-dir experiments/atb_validation_v1/exp11_rsm_residual_stream_memory/run_TIMESTAMP
```

## Arms

| Variant | Description |
| --- | --- |
| `base_model` | No injection. |
| `correct_memory` | Current fact memory plus distractors. |
| `random_memory` | Unrelated fact memories only. |
| `shuffled_layers` | Correct memory bank with layer axis permuted. |
| `gate_off` | Correct memory bank, all memories injected with unit score. |

Primary score:

```text
gap = margin(correct_memory) - max(random_memory, shuffled_layers, gate_off)
```

