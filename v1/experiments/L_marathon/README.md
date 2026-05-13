# L Marathon: Long-Conversation Bank Stability

This experiment tests bank injection stability across N turns of dialogue. See `PREREG.md` for the full specification.

## Quick start

```bash
python -m experiments.L_marathon.run \
  --model gpt2-medium \
  --method caa \
  --device cpu \
  --dtype fp32 \
  --seed 0 \
  --turns 100 \
  --inject-facts experiments/L_marathon/facts_3.jsonl \
  --probe-set experiments/L_marathon/probes_8.jsonl \
  --filler experiments/L_marathon/filler.txt \
  --out runs/L_marathon/run_001/ \
  --resume
```

Run `python -m experiments.L_marathon.run --help` for all options.

## Output

- `cells.jsonl` — one row per probe checkpoint (turn)
- `env.json` — environment metadata
- `summary.json` — H_L verdicts per model (produced by `aggregate.py`)

Author: BIRI GA, 2026-05-10.
