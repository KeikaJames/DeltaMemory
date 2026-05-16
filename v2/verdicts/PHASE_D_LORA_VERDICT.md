# Phase-D LoRA / Adapter Verdict

**Status**: PASS as a falsifying ablation for content-bank specificity.

**Purpose**: Test whether early v2 train/test NLL lift requires a content-addressed bank, or whether a small trainable residual module can absorb the same split under a matched parameter budget.

**Common config**: Qwen/Qwen3-4B-Instruct-2507, layer 9, frozen base model, 200 training steps, 120 train / 120 eval facts, rank 64, MPS bf16 loading. All runs uninstall the adapter after evaluation and verify that test NLL returns to base.

| Method | Trainable params | Seeds | Mean test NLL before | Mean test NLL after | Mean Delta NLL |
|---|---:|---:|---:|---:|---:|
| plain_adapter | 327,680 | 0,1,2 | 11.981 | 4.353 | -7.628 |
| lora_q | 425,984 | 0,1,2 | 11.981 | 11.760 | -0.221 |
| lora_qk | 327,680 | 0,1,2 | 11.981 | 11.736 | -0.246 |

**Interpretation**: The plain residual adapter is a strong positive PEFT baseline on the same data split, while Q/K LoRA at this single layer is weak. This confirms that a small trainable module can create large NLL lift without any content-bearing bank rows. The result strengthens the E10/E11/E20C conclusion: multi-slot v2 lift should not be interpreted as factual memory retrieval unless item-specific controls also pass.

**Paper-facing claim allowed**: v2 contains a trainable activation-side adapter path capable of large supervised NLL reduction. It does not establish scalable external memory.

**Paper-facing claim disallowed**: A 512-slot bank stores fact identity or performs reliable content-addressed recall.

**Evidence files**:

- `v2/experiments/e_phase_d_lora/phase_d_lora_plain_adapter_seed0_r64.json`
- `v2/experiments/e_phase_d_lora/phase_d_lora_plain_adapter_seed1_r64.json`
- `v2/experiments/e_phase_d_lora/phase_d_lora_plain_adapter_seed2_r64.json`
- `v2/experiments/e_phase_d_lora/phase_d_lora_lora_q_seed0_r64.json`
- `v2/experiments/e_phase_d_lora/phase_d_lora_lora_q_seed1_r64.json`
- `v2/experiments/e_phase_d_lora/phase_d_lora_lora_q_seed2_r64.json`
- `v2/experiments/e_phase_d_lora/phase_d_lora_lora_qk_seed0_r64.json`
- `v2/experiments/e_phase_d_lora/phase_d_lora_lora_qk_seed1_r64.json`
- `v2/experiments/e_phase_d_lora/phase_d_lora_lora_qk_seed2_r64.json`

Run `python3 v2/scripts/prepublish_audit.py` before citing these numbers.
