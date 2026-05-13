# Makefile for MnEmE/RCV-HC
# Usage: make <target>

.PHONY: help test lint runs-index check-auth

help:
	@echo "Available targets:"
	@echo "  make test          — run pytest (CPU-only, no GPU required)"
	@echo "  make lint          — run ruff linter"
	@echo "  make runs-index    — regenerate tracked runs/INDEX.md (local archive: pass --local manually)"
	@echo "  make check-auth    — check tracked run dirs for cells.jsonl + env.json (strict)"

test:
	pytest tests/ -x -q

lint:
	ruff check deltamemory/ tests/ scripts/ integrations/ examples/

runs-index:
	python3 scripts/regen_runs_index.py

check-auth:
	python3 scripts/regen_runs_index.py --check-only

.PHONY: docker-build

docker-build:
	@if ! command -v docker >/dev/null 2>&1; then echo "SKIP: docker not available; install Docker/Buildx to build docker/Dockerfile.production"; elif ! docker info >/dev/null 2>&1; then echo "SKIP: docker daemon unavailable or permission denied; not building docker/Dockerfile.production"; else docker buildx build --platform $${PLATFORMS:-linux/amd64,linux/arm64} --output $${OUTPUT:-type=cacheonly} -f docker/Dockerfile.production -t mneme:production .; fi
