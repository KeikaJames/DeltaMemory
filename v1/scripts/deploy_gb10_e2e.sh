#!/usr/bin/env bash
# deploy_gb10_e2e.sh — GB10 end-to-end deployment witness for Mneme v0.6
#
# What it does:
#   1. Loads gemma-4-31B-it on spark1 (via uvicorn + deltamemory.api.app)
#   2. Starts the FastAPI app on port 8741
#   3. Writes a 100-fact bank via POST /bank/write
#   4. Sends 100 fact-recall HTTP requests via Python httpx
#   5. Logs per-request: latency_ms, recall@5 hit (target in top-5)
#   6. Tracks RSS memory across the run (via psutil)
#   7. Outputs runs/GB10_e2e_v1/REPORT.md with stats
#
# Usage (on spark1):
#   cd /home/gabira/projects/RCV-HC
#   source .venv-gb10/bin/activate
#   bash scripts/deploy_gb10_e2e.sh
#
# Environment overrides:
#   MODEL_PATH    — default: /home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it
#   PORT          — default: 8741
#   N_FACTS       — default: 100
#   N_REQUESTS    — default: 100
#   ALPHA         — default: 1.0
#   OUT_DIR       — default: runs/GB10_e2e_v1
#
set -euo pipefail

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/home/gabira/Desktop/workspace/models/whitelist/gemma-4-31B-it}"
PORT="${PORT:-8741}"
N_FACTS="${N_FACTS:-100}"
N_REQUESTS="${N_REQUESTS:-100}"
ALPHA="${ALPHA:-1.0}"
OUT_DIR="${OUT_DIR:-runs/GB10_e2e_v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "==============================="
echo " Mneme GB10 E2E Deployment"
echo " $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==============================="
echo "MODEL_PATH : $MODEL_PATH"
echo "PORT       : $PORT"
echo "N_FACTS    : $N_FACTS"
echo "N_REQUESTS : $N_REQUESTS"
echo "ALPHA      : $ALPHA"
echo "OUT_DIR    : $OUT_DIR"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH not found: $MODEL_PATH"
    exit 1
fi

# --------------------------------------------------------------------------
# 1. Generate 100 synthetic facts
# --------------------------------------------------------------------------
FACTS_FILE="$OUT_DIR/e2e_facts.jsonl"
echo "[1/6] Generating $N_FACTS synthetic facts → $FACTS_FILE"
python3 - <<'PYEOF'
import json, os, sys
n = int(os.environ.get("N_FACTS", "100"))
out = os.environ.get("FACTS_FILE", "runs/GB10_e2e_v1/e2e_facts.jsonl")
rows = []
for i in range(n):
    subj = f"Entity{i:04d}"
    val  = f"Value{i:04d}"
    rows.append({
        "fact_id":      f"fact_{i:04d}",
        "write_prompt": f"The identifier of {subj} is {val}.",
        "address":      subj,
        "target":       val,
        "probe":        f"The identifier of {subj} is",
    })
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"  wrote {n} facts to {out}")
PYEOF
export FACTS_FILE

# --------------------------------------------------------------------------
# 2. Start the FastAPI server
# --------------------------------------------------------------------------
echo "[2/6] Starting FastAPI server on port $PORT …"
MNEME_MODEL_PATH="$MODEL_PATH" \
MNEME_DEVICE=cuda \
MNEME_DTYPE=bfloat16 \
MNEME_ALPHA="$ALPHA" \
    uvicorn deltamemory.api.app:app \
        --host 0.0.0.0 \
        --port "$PORT" \
        --log-level warning &
SERVER_PID=$!
echo "  server PID=$SERVER_PID"

# Wait for the server to become healthy (model load can take ~2 min)
echo "  waiting for server to be healthy …"
for i in $(seq 1 120); do
    if curl -sf "http://localhost:${PORT}/health" | grep -q '"status":"ok"'; then
        echo "  healthy after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server process died"
        exit 1
    fi
    sleep 1
done

# Check model is loaded
curl -sf "http://localhost:${PORT}/health" | python3 -c \
    "import json,sys; d=json.load(sys.stdin); assert d['model_loaded'], 'model not loaded'"

# --------------------------------------------------------------------------
# 3. Write 100-fact bank
# --------------------------------------------------------------------------
echo "[3/6] Writing $N_FACTS facts to the bank …"
python3 - <<'PYEOF'
import json, os, httpx, sys
port  = os.environ.get("PORT", "8741")
fname = os.environ.get("FACTS_FILE", "runs/GB10_e2e_v1/e2e_facts.jsonl")
with open(fname) as f:
    rows = [json.loads(l) for l in f if l.strip()]
facts = [{"fact_id": r["fact_id"], "write_prompt": r["write_prompt"],
          "address": r["address"], "policy": "period"} for r in rows]
# Write in batches of 10
batch_size = 10
for i in range(0, len(facts), batch_size):
    batch = facts[i:i+batch_size]
    r = httpx.post(f"http://localhost:{port}/bank/write",
                   json={"facts": batch}, timeout=120.0)
    r.raise_for_status()
    d = r.json()
    print(f"  batch {i//batch_size+1}: total_facts={d['total_facts']}")
print("  bank write complete")
PYEOF

# Verify bank status
curl -sf "http://localhost:${PORT}/bank/status" | python3 -c \
    "import json,sys; d=json.load(sys.stdin); print(f\"  bank: {d['num_facts']} facts\")"

# --------------------------------------------------------------------------
# 4. Send 100 fact-recall requests + measure latency & recall@5
# --------------------------------------------------------------------------
echo "[4/6] Sending $N_REQUESTS recall requests …"
RESULTS_FILE="$OUT_DIR/request_results.jsonl"
python3 - <<'PYEOF'
import json, os, time, httpx, sys
from pathlib import Path

port     = os.environ.get("PORT", "8741")
n_req    = int(os.environ.get("N_REQUESTS", "100"))
alpha    = float(os.environ.get("ALPHA", "1.0"))
fname    = os.environ.get("FACTS_FILE", "runs/GB10_e2e_v1/e2e_facts.jsonl")
out_file = os.environ.get("RESULTS_FILE", "runs/GB10_e2e_v1/request_results.jsonl")

with open(fname) as f:
    rows = [json.loads(l) for l in f if l.strip()][:n_req]

results = []
with httpx.Client(timeout=60.0) as client:
    for i, row in enumerate(rows):
        payload = {
            "prompt":         row["probe"],
            "max_new_tokens": 8,
            "alpha":          alpha,
            "temperature":    0.0,
        }
        t0 = time.perf_counter()
        r  = client.post(f"http://localhost:{port}/generate", json=payload)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        r.raise_for_status()
        d = r.json()
        hit = any(row["target"].lower() in t.lower() for t in d["recall_top5"])
        record = {
            "idx":          i,
            "fact_id":      row["fact_id"],
            "target":       row["target"],
            "top5":         d["recall_top5"],
            "hit":          hit,
            "latency_ms":   d["latency_ms"],
            "wall_ms":      wall_ms,
        }
        results.append(record)
        if (i+1) % 10 == 0:
            hits_so_far = sum(x["hit"] for x in results)
            print(f"  {i+1}/{n_req}  recall@5={hits_so_far/(i+1):.3f}  "
                  f"p50_lat={sorted(x['latency_ms'] for x in results)[len(results)//2]:.0f}ms")

with open(out_file, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"  wrote {len(results)} results → {out_file}")
PYEOF

# --------------------------------------------------------------------------
# 5. Track peak RSS
# --------------------------------------------------------------------------
echo "[5/6] Sampling RSS of server process …"
PEAK_RSS_MB=$(python3 - <<PYEOF
import os, sys
try:
    import psutil
    p = psutil.Process(int("$SERVER_PID"))
    rss = p.memory_info().rss / 1024**2
    print(f"{rss:.0f}")
except Exception as e:
    print("N/A")
PYEOF
)
echo "  peak RSS ≈ ${PEAK_RSS_MB} MB"

# --------------------------------------------------------------------------
# 6. Stop server + generate report
# --------------------------------------------------------------------------
echo "[6/6] Stopping server and generating report …"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# Write env.json
python3 - <<'PYEOF'
import json, os, sys, datetime, socket, subprocess, hashlib, platform
from pathlib import Path

out_dir = os.environ.get("OUT_DIR", "runs/GB10_e2e_v1")
model   = os.environ.get("MODEL_PATH", "unknown")

def _git(*args):
    try: return subprocess.check_output(["git"]+list(args), stderr=subprocess.DEVNULL).decode().strip()
    except: return ""

env = {
    "commit": _git("rev-parse","HEAD"),
    "dirty":  bool(_git("status","--porcelain")),
    "model_path": model,
    "device": "cuda",
    "dtype": "bfloat16",
    "host": socket.gethostname(),
    "started_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "python": platform.python_version(),
}
try:
    import torch; env["torch"] = torch.__version__
    env["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
except: pass
try:
    import transformers; env["transformers"] = transformers.__version__
except: pass

Path(out_dir, "env.json").write_text(json.dumps(env, indent=2, sort_keys=True)+"\n")
print("  wrote env.json")
PYEOF

# Generate REPORT.md from results
python3 - <<'PYEOF'
import json, os, statistics, math
from pathlib import Path

out_dir  = os.environ.get("OUT_DIR", "runs/GB10_e2e_v1")
res_file = os.path.join(out_dir, "request_results.jsonl")
n_req    = int(os.environ.get("N_REQUESTS","100"))
n_facts  = int(os.environ.get("N_FACTS","100"))
alpha    = os.environ.get("ALPHA","1.0")
model    = os.environ.get("MODEL_PATH","unknown")
peak_rss = os.environ.get("PEAK_RSS_MB","N/A")

if not Path(res_file).exists():
    print("  no results file — report skipped")
    raise SystemExit(0)

with open(res_file) as f:
    rows = [json.loads(l) for l in f if l.strip()]

lats  = sorted(r["latency_ms"] for r in rows)
hits  = [r["hit"] for r in rows]
n     = len(rows)

p50 = lats[n//2]
p95 = lats[int(n*0.95)]
p99 = lats[int(n*0.99)] if n >= 100 else lats[-1]
recall5 = sum(hits)/n if n else 0.0

report = f"""# GB10 End-to-End Deployment Witness — v1

**Date**: {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d')}
**Model**: `{model}`
**Host**: {__import__('socket').gethostname()}
**Device**: CUDA / bfloat16
**Alpha**: {alpha}
**Bank size**: {n_facts} facts

---

## Results

| Metric | Value |
|---|---|
| Total requests | {n} |
| Recall@5 | **{recall5:.3f}** ({sum(hits)}/{n} hits) |
| Latency p50 | {p50:.1f} ms |
| Latency p95 | {p95:.1f} ms |
| Latency p99 | {p99:.1f} ms |
| Peak RSS | {peak_rss} MB |

---

## Methodology

- 100 synthetic facts written: `Entity{i:04d} → Value{i:04d}`
- Recall probe: `"The identifier of EntityXXXX is"` → target `ValueXXXX`
- Hit criterion: target token appears in top-5 logits at last prompt position
- Latency: time from POST /generate to response (includes server-side generate)
- RSS: sampled from server PID via psutil at end of request loop

---

## Raw data

`request_results.jsonl` — {n} rows with `fact_id, target, top5, hit, latency_ms`.
"""

Path(out_dir, "REPORT.md").write_text(report)
print(f"  recall@5={recall5:.3f}  p50={p50:.0f}ms  p95={p95:.0f}ms  RSS={peak_rss}MB")
print(f"  wrote {out_dir}/REPORT.md")
PYEOF

echo ""
echo "============================="
echo " GB10 E2E complete"
echo " Report: $OUT_DIR/REPORT.md"
echo "============================="
