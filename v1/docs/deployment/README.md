# Deployment Guides

Deployment documentation for the Mneme delta-memory library.

| Sub-page | Task | Status |
|---|---|---|
| [D.1 — GB10 on-device deployment](d1-gb10.md) | Apple GB10 / Neural Engine optimisation | (planned) |
| [D.3 — Spark batch inference](d3-spark.md) | PySpark integration for large-scale offline injection | (planned) |
| [D.4 — Docker (CPU + CUDA)](docker.md) | Multistage production Dockerfile | ✅ shipped v0.5 |

## Roadmap

- **I.1 — FastAPI hardening:** `/healthz` liveness probe, `/metrics`
  Prometheus endpoint, graceful shutdown. Completes the healthcheck wired
  into the D.4 Dockerfile.
