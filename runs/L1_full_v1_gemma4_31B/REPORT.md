# L.1 Marathon — gemma-4-31B-it — PENDING

**Status**: Awaiting spark1 run.

## Setup

- Model: gemma-4-31B-it
- Method: lopi_default
- Seeds: 0, 1, 2
- Turns: 2000

## Coordination

Check GPU availability before launch:
```bash
ssh spark1 "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
```

Run after Track A is between models (GPU contention window).

## Reproduction

```bash
ssh spark1
cd /home/gabira/projects/RCV-HC && source .venv-gb10/bin/activate
METHODS=lopi_default SEEDS="0 1 2" TURNS=2000 bash scripts/dispatch_L1_gemma_only.sh
```
