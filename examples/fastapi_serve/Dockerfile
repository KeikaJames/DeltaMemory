# =============================================================================
# Mneme delta-memory  —  multistage production Dockerfile
#
# Build targets:
#   docker buildx build --target runtime-cpu  -t mneme:cpu  examples/fastapi_serve
#   docker buildx build --target runtime-cuda -t mneme:cuda examples/fastapi_serve
#
# MPS note: Apple Metal Performance Shaders are not accessible inside Docker
# containers. For MPS development, use the `runtime-cpu` image and run on the
# host with:  MNEME_DEVICE=mps uvicorn examples.fastapi_serve.app:app ...
#
# Runtime environment variables:
#   MNEME_DEVICE   — device hint passed to the app (cpu | cuda | mps)
#   MNEME_DTYPE    — torch dtype override (float32 | float16 | bfloat16)
#   HF_HOME        — HuggingFace cache root (mount a volume here)
#   MNEME_MODEL_ID — model to serve (default: Qwen/Qwen2.5-0.5B)
# =============================================================================

ARG TARGET=cpu

# -----------------------------------------------------------------------------
# Stage 1 — builder
# Python 3.12 chosen because PyTorch CPU/CUDA wheels are most stable there;
# the slim base keeps layer size small while build-essential + git are only
# needed to compile any C-extension deps at install time.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the files needed to install the package so this layer is
# invalidated only when they change, not on every source edit.
COPY pyproject.toml .
COPY deltamemory/ deltamemory/
COPY examples/fastapi_serve/ examples/fastapi_serve/

# Install into --user prefix so the runtime stage can copy just /root/.local.
# BuildKit pip cache avoids re-downloading wheels on incremental rebuilds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --user ".[test]" \
    && pip install --no-cache-dir --user fastapi "uvicorn[standard]"

# -----------------------------------------------------------------------------
# Stage 2 — runtime-cpu
# Minimal Python 3.12-slim; no CUDA runtime.  All packages come from builder.
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime-cpu

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MNEME_DEVICE=cpu

# Copy the pre-built user-site from builder
COPY --from=builder /root/.local /root/.local

# Copy application source (needed so `examples.fastapi_serve.app` is importable)
WORKDIR /app
COPY --from=builder /build /app

# Non-root runtime user
RUN useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin mneme \
    && chown -R mneme:mneme /app /root/.local
USER mneme

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "examples.fastapi_serve.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# -----------------------------------------------------------------------------
# Stage 3 — runtime-cuda
# CUDA 12.4.1 runtime on Ubuntu 22.04.  Python 3.12 installed via deadsnakes.
# MPS is not Docker-accessible; see the header note above.
# CUDA torch wheels are used automatically when the CUDA runtime is present.
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-cuda

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-distutils \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python

ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MNEME_DEVICE=cuda

COPY --from=builder /root/.local /root/.local

WORKDIR /app
COPY --from=builder /build /app

RUN useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin mneme \
    && chown -R mneme:mneme /app /root/.local
USER mneme

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uvicorn", "examples.fastapi_serve.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
