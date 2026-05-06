#!/usr/bin/env bash
# Sync downloaded whitelist models from local Mac → spark1 (air-gapped via SSH).
# Run from /Users/gabiri/projects/RCV-HC after downloads complete.
set -euo pipefail

REMOTE=spark1
DEST=/home/gabira/Desktop/workspace/models/whitelist
SRC=/Users/gabiri/models_staging

for m in gemma-4-31B-it Qwen3.6-27B gpt-oss-120b; do
    if [ ! -d "$SRC/$m" ]; then
        echo "[skip] $m not staged"
        continue
    fi
    echo "=== rsync $m ==="
    rsync -avh --progress --partial \
        "$SRC/$m/" "$REMOTE:$DEST/$m/"
done

echo "=== verify on spark1 ==="
ssh $REMOTE "du -sh $DEST/*"
