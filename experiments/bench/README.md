# Mneme injector benchmark harness

`run_bench.py` measures prefill+one-token decode latency, tokens/sec, peak VRAM
where available, and process RSS for `none`, `caa`, `scar`, and `lopi` modes.

The only run validated locally is the no-download smoke path:

```bash
python experiments/bench/run_bench.py --smoke --inject caa \
  --iters 3 --out experiments/bench/results/smoke.jsonl
```

GB10 runs should be executed from `~/desktop/workspace/` with existing mounted
Hugging Face weights/cache. The harness passes `local_files_only=True` for
non-smoke model loading, so missing weights fail fast instead of downloading.

Example:

```bash
cd ~/desktop/workspace/RCV-HC
source .venv-gb10/bin/activate
python experiments/bench/run_bench.py \
  --model /path/to/mounted/Qwen3-7B \
  --backend cuda --dtype bf16 --inject scar \
  --batch 4 --seq 2048 --warmup 5 --iters 20 \
  --out experiments/bench/results/qwen3_7b_scar_b4_t2048.jsonl
```

The YAML files in `experiments/bench/configs/` document the intended GB10 sweep:
Qwen3-7B, Gemma3-12B, and Llama3-8B over batch `[1, 4, 16]`, sequence
`[512, 2048, 8192]`, and all four injection modes.
