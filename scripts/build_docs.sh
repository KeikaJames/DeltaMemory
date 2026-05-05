#!/usr/bin/env bash
set -euo pipefail
if ! command -v pdoc >/dev/null 2>&1; then echo "pdoc is not installed; skipping API docs. Install with: pip install -e '.[docs]'"; exit 0; fi
mkdir -p docs/api
pdoc -o docs/api/ deltamemory
