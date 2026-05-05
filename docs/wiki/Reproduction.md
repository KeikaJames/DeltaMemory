# Reproduction

This page documents how to reproduce every published number in the repo, top-down from `git clone` to the final tables.

## Environment

| | Mac (development) | NVIDIA GB10 "spark" (compute) |
|---|---|---|
| OS | macOS (Apple Silicon) | Ubuntu 22.04 |
| Python | 3.11 | 3.12 |
| torch | 2.11.0 (MPS / Metal) | 2.10.0+cu13 |
| transformers | 4.46+ | 5.5.0 |
| dtype | bfloat16 | bfloat16 |

The Apple Silicon path is documented in `docs/apple_silicon.md`. The CUDA path uses cuda capability 12.1 (Blackwell); PyTorch 2.10 emits a benign capability warning.

## 1. Clone + install

```bash
git clone https://github.com/KeikaJames/MnEmE.git
cd Mneme
python3.11 -m venv .venv-mac
source .venv-mac/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Conservation law (the red line)

This is the single most important test. If it fails, **stop**.

```bash
# Mac (Gemma-4)
python tests/conservation_real_models.py --models gemma-4-E2B --device mps

# GB10 (Gemma-4 + Qwen3)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python tests/conservation_real_models.py \
    --models gemma-4-E2B qwen3-4b --device cuda
```

Expected output:

```
=== google/gemma-4-E2B ===
  adapter = gemma4, num_layers = 35
  max-abs-diff = 0.000e+00  mean-abs-diff = 0.000e+00  bit_equal = True
=== Qwen/Qwen3-4B-Instruct-2507 ===
  adapter = qwen3, num_layers = 36
  max-abs-diff = 0.000e+00  mean-abs-diff = 0.000e+00  bit_equal = True
```

## 3. Unit tests

```bash
python -m pytest tests/ -q
```

Expected: all local tests pass. The exact count changes as regression tests
are added; the heavyweight real-model conservation suite is opt-in.

## 4. Phase N intervention demo (qualitative)

```bash
# Mac
python scripts/run_intervention_demo.py \
    --model google/gemma-4-E2B --device mps --dtype bfloat16 --alpha 1.0

# GB10
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/run_intervention_demo.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --device cuda --dtype bfloat16 --alpha 1.0
```

Outputs:
- `transcripts/v3_intervention/<model>/demo.md` — markdown with per-fact tables
- `transcripts/v3_intervention/<model>/demo.json` — raw numbers

The committed transcripts in this repo are the **canonical** output; if your numbers differ, the run is non-canonical (different transformer version, dtype, or device).

## 5. Phase G test eval (the honest negative)

Reproduces `reports/cleanroom/stage14_test_gemma4_e2b/REPORT.md` — the held-out
test eval that **rejected H1** (v3 = 0.278 < B0 = 0.359, p = 0.007). The
methodology amendment in that report describes Phase G+1 (Stage 15) corrections.
Run via the per-script Stage 14 entrypoints under
`scripts/legacy/run_stage14_*.py` (see `scripts/legacy/README.md`).

## 6. Phase M cross-arch benchmark (v3.1)

```bash
python scripts/run_v31_benchmark.py --help
```

Generates `reports/cleanroom/stage15_v31_test/<model>/REPORT.md` per
architecture, with the same B0 / B1 / B3 (MEMIT) / B4 (LoRA) baselines.

## 7. Determinism notes

- All eval scripts seed Python / NumPy / Torch via `--seed`. Default: 0.
- bf16 has limited precision, so we report Wilcoxon paired ranks rather than mean diff with a tolerance.
- The `bit_equal` test is a **logits** comparison, not a sampled-token comparison; sampling is non-deterministic on MPS even with `set_seed`.

## 8. Hardware shopping list (if you want to reproduce on your own box)

| stage | min compute | recommended |
|---|---|---|
| Conservation test | any | Mac M-series with 16 GB RAM |
| Phase N demo (1 model) | 24 GB GPU or 32 GB unified | NVIDIA RTX 4090 / Apple M3 Pro |
| Phase G eval (Gemma-4-E2B) | 32 GB unified or 24 GB GPU | Apple M3 Max / RTX 4090 |
| Phase L (training v3.1 K-projector) | 80 GB GPU or GB10 | NVIDIA H100 / GB10 |
| Phase M (DeepSeek-32B bench) | 80 GB GPU | GB10 |

## 9. Known gotchas

- **`HF_HUB_OFFLINE=1`** is required when running on a sandboxed GB10 with no internet. Models must be pre-downloaded.
- **`attn_implementation="eager"`** is required — Mneme patches the eager forward. Flash-attn / SDPA paths will silently bypass the bank.
- **`use_cache=False`** during writes (already set in `write_fact`). With KV-cache enabled the capture position is wrong on subsequent forwards.
- **bf16 vs fp16**: results are stable in bf16, less so in fp16 (we observed up to 0.04 logprob drift). Prefer bf16.

## 10. If something fails

1. Re-run the conservation test in isolation. If it fails, the core invariant is broken; stop and bisect.
2. Check `git status` — modified files outside `transcripts/` and `reports/` may indicate uncommitted local changes affecting the run.
3. Re-pull and re-run; the canonical numbers are pinned to commit SHAs in `reports/cleanroom/*/REPORT.md`.
