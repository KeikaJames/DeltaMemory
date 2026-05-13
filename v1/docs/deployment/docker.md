# D.4 — Production Docker Deployment

> **Status:** shipped in v0.5 `feat/v05-counterfactual-industrial`.
> **Next:** I.1 — FastAPI hardening (`/metrics`, `/healthz` endpoints).

## Overview

`examples/fastapi_serve/Dockerfile` is a three-stage BuildKit image:

| Stage | Base | Purpose |
|---|---|---|
| `builder` | `python:3.12-slim` | Install all Python deps into `--user` prefix |
| `runtime-cpu` | `python:3.12-slim` | CPU-only production image |
| `runtime-cuda` | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | GPU production image |

Python 3.12 is used because PyTorch CPU and CUDA wheels are most stable on
that release.

## Building

> Commands are run from the **repository root**.

```bash
# CPU image
docker buildx build --target runtime-cpu \
  -t mneme:cpu examples/fastapi_serve

# CUDA image
docker buildx build --target runtime-cuda \
  -t mneme:cuda examples/fastapi_serve

# Add --no-cache --progress=plain for full build output
```

## Running

```bash
# Basic CPU run
docker run --rm -p 8000:8000 mneme:cpu

# CUDA run (requires NVIDIA Container Toolkit)
docker run --rm --gpus all -p 8000:8000 mneme:cuda

# Mount a persistent HuggingFace cache and memory bank
docker run --rm \
  -v /var/lib/mneme/hf_cache:/root/.cache/huggingface \
  -v /var/lib/mneme/bank:/data/bank \
  -e HF_HOME=/root/.cache/huggingface \
  -e MNEME_BANK_PATH=/data/bank \
  -p 8000:8000 mneme:cpu
```

## Runtime Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MNEME_DEVICE` | `cpu` / `cuda` (per image) | Device hint (`cpu`, `cuda`, `mps`*) |
| `MNEME_DTYPE` | — | torch dtype override (`float32`, `float16`, `bfloat16`) |
| `MNEME_MODEL_ID` | `Qwen/Qwen2.5-0.5B` | HuggingFace model identifier |
| `HF_HOME` | system default | HuggingFace cache root — mount a volume here |
| `MNEME_BANK_PATH` | — | Path to the persistent memory bank directory |

*MPS (Apple Metal) is **not accessible inside Docker containers**. For MPS
development, use the `runtime-cpu` image with the flag `MNEME_DEVICE=mps`
and run the server directly on the host:

```bash
MNEME_DEVICE=mps uvicorn examples.fastapi_serve.app:app \
  --host 0.0.0.0 --port 8000
```

## Persistent Bank Volume

Mount a host directory to persist the memory bank across container restarts:

```bash
docker run --rm \
  -v /var/lib/mneme/bank:/data/bank \
  -e MNEME_BANK_PATH=/data/bank \
  -p 8000:8000 mneme:cpu
```

Use a named Docker volume for managed lifecycle:

```bash
docker volume create mneme-bank
docker run --rm \
  -v mneme-bank:/data/bank \
  -e MNEME_BANK_PATH=/data/bank \
  -p 8000:8000 mneme:cpu
```

## Security Defaults

- **Non-root user:** the container runs as `mneme` (uid 1000). No capability
  escalation is needed.
- **No Python bytecode:** `PYTHONDONTWRITEBYTECODE=1` prevents `.pyc` files
  being written into read-only layers.
- **Unbuffered stdout/stderr:** `PYTHONUNBUFFERED=1` ensures log lines are
  flushed immediately to container stdout for log aggregators.
- **Healthcheck:** the runtime stages declare:
  ```
  HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3
      CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
  ```
  The `/healthz` endpoint is added in **I.1** (FastAPI hardening). Until then,
  Docker marks the container as `unhealthy` after `start-period` but the
  service continues to run.

## Build Context

`.dockerignore` caps the build context well under 50 MB by excluding:
`__pycache__/`, `.git/`, `.venv*/`, `tests/`, `experiments/`, `reports/`,
`transcripts/`, `eval/`, `docs/figures/`, and common binary/temp files.

## Legacy Image

`examples/fastapi_serve/Dockerfile.toy` is the original single-stage file
kept for backward-compatible documentation references. Do not use it in
production.
