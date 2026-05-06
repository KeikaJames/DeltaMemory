# B2 Sparsity Test — PENDING

**Status**: Awaiting spark1 run.

## Reproduction

```bash
ssh spark1
cd /home/gabira/projects/RCV-HC && source .venv-gb10/bin/activate
python experiments/X7_mech/sparsity_test.py \
  --model /home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it \
  --device cuda --dtype bf16 \
  --out runs/X7_mech_v1_b2
python experiments/X7_mech/aggregate.py --run-dir runs/X7_mech_v1_b2 --sub B2
```
