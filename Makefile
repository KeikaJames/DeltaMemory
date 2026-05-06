# Makefile for MnEmE/RCV-HC
# Usage: make <target>

.PHONY: help test lint runs-index check-auth

help:
	@echo "Available targets:"
	@echo "  make test          — run pytest (CPU-only, no GPU required)"
	@echo "  make lint          — run ruff linter"
	@echo "  make runs-index    — regenerate runs/INDEX.md"
	@echo "  make check-auth    — check all run dirs for cells.jsonl + env.json (strict)"

test:
	pytest tests/ -x -q

lint:
	ruff check deltamemory/ tests/ scripts/ integrations/ examples/

runs-index:
	python3 scripts/regen_runs_index.py

check-auth:
	python3 scripts/regen_runs_index.py --check-only
