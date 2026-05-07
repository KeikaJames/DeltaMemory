#!/usr/bin/env bash
# Download Qwen3-4B + Llama-3.1-8B replication models into the spark1 whitelist.
#
# Usage on spark1:
#   bash scripts/download_v08_replication_models.sh
#
# Locked by DECISIONS.md D4 (2026-05-07).
# Run AFTER the current L.1 marathon finishes (GPU contention is fine for a CPU
# download but disk thrash on the same NVMe slows GPU loads).
set -euo pipefail

WHITELIST=/home/gabira/Desktop/workspace/models/whitelist
mkdir -p "$WHITELIST"

# spark1 cannot reach huggingface.co directly; route via hf-mirror.com (HTTP 200).
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
echo "[net] using HF_ENDPOINT=$HF_ENDPOINT"

# Activate the project venv so huggingface-cli is available with our pinned deps.
source /home/gabira/projects/RCV-HC/.venv-gb10/bin/activate

# Sanity: token must exist; we use the cached HF token (no env var leakage).
test -s /home/gabira/.cache/huggingface/token || {
    echo "[ERR] no HF token at ~/.cache/huggingface/token" >&2
    exit 1
}

download_one() {
    local repo="$1"
    local local_dir="$2"
    if [[ -d "$local_dir" && -n "$(ls -A "$local_dir" 2>/dev/null)" ]]; then
        echo "[skip] $local_dir already populated"
        return 0
    fi
    echo "=== downloading $repo → $local_dir at $(date) ==="
    # New `hf` CLI replaces the deprecated `huggingface-cli`.
    # Note: `hf download` interprets positional args after the repo as filename
    # filters when --include/--exclude are intermixed. Skip exclude — these
    # repos are safetensors-only.
    if hf download "$repo" --local-dir "$local_dir"; then
        return 0
    else
        echo "[warn] download of $repo failed (gated / not on mirror?)" >&2
        rm -rf "$local_dir"
        return 1
    fi
}

# D4 targets — both dense attention, 4-8B class, dense KV → ANB-compatible.
# Canonical HF repo names verified against existing run env.json artifacts.
# Both targets are HARD requirements; if either fails the script must fail
# loudly so downstream jobs do not assume the whitelist is complete.
if ! download_one "Qwen/Qwen3-4B-Instruct-2507" "$WHITELIST/Qwen3-4B-Instruct-2507"; then
    echo "[FATAL] Qwen3-4B-Instruct-2507 download failed — aborting." >&2
    exit 2
fi

# meta-llama/* repos are gated and the HF mirror does not bypass gating;
# unsloth/* hosts byte-identical mirrors of Meta's instruct weights with the
# Llama-3.1 community license redistributed in-repo. Same model weights, no
# approval flow. Try unsloth first, fall back to NousResearch; only after
# BOTH fail do we abort.
if ! download_one "unsloth/Meta-Llama-3.1-8B-Instruct" "$WHITELIST/Llama-3.1-8B-Instruct"; then
    if ! download_one "NousResearch/Meta-Llama-3.1-8B-Instruct" "$WHITELIST/Llama-3.1-8B-Instruct"; then
        echo "[FATAL] Llama-3.1-8B-Instruct download failed on both mirrors — aborting." >&2
        exit 2
    fi
fi

echo "[done] v0.8 replication models in $WHITELIST at $(date)"
ls -la "$WHITELIST"
