#!/usr/bin/env python3
"""A4 derivative-gate metric probe.

Measures three adjacent-token change metrics on a long prompt with a known
math→cooking boundary: current L2(Q_t-Q_{t-1}), cosine distance on Q, and
cosine distance on residual hidden states.  Writes raw scores and simple ROC AUC.
"""
from __future__ import annotations

import argparse, json, platform, subprocess
from pathlib import Path

import torch
import torch.nn.functional as F


def _env(dtype: str, device: str) -> dict:
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "dtype": dtype, "device": device, "python": platform.python_version(), "mps_available": torch.backends.mps.is_available()}


def _auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            wins += p > n
            ties += p == n
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _prompt() -> str:
    math = " ".join(["Let x be an integer and solve the quadratic equation by completing the square."] * 28)
    cooking = " ".join(["Chop onions, warm olive oil, simmer tomatoes, and season the soup carefully."] * 28)
    return math + " BOUNDARY " + cooking


def run(args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=True, attn_implementation="eager", low_cpu_mem_usage=True, local_files_only=args.local_files_only).to(args.device)
    model.eval()
    text = _prompt()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(args.device)
    boundary_ids = tok(text.split(" BOUNDARY ")[0] + " BOUNDARY", return_tensors="pt", truncation=True, max_length=args.max_length).input_ids
    boundary = min(boundary_ids.shape[1] - 1, enc.input_ids.shape[1] - 2)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
    h = out.hidden_states[args.layer][0].float()  # (T,D)
    # Locate q_proj from the same layer where possible.
    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else model.model.model.layers
    attn = layers[args.layer].self_attn
    with torch.no_grad():
        q = attn.q_proj(out.hidden_states[args.layer][0]).float()
    T = h.shape[0]
    q = q.reshape(T, -1)
    l2 = torch.linalg.vector_norm(q[1:] - q[:-1], dim=-1)
    cos_q = 1.0 - F.cosine_similarity(q[1:], q[:-1], dim=-1)
    cos_h = 1.0 - F.cosine_similarity(h[1:], h[:-1], dim=-1)
    labels = [abs((i + 1) - boundary) <= args.positive_window for i in range(T - 1)]
    rows = []
    for i in range(T - 1):
        rows.append({"edge": [i, i+1], "is_boundary": labels[i], "l2_q": float(l2[i]), "cos_q": float(cos_q[i]), "cos_h": float(cos_h[i])})
    metrics = {"l2_q_auc": _auc([r["l2_q"] for r in rows], labels), "cos_q_auc": _auc([r["cos_q"] for r in rows], labels), "cos_h_auc": _auc([r["cos_h"] for r in rows], labels)}
    return {"env": _env(args.dtype, args.device), "model": args.model, "layer": args.layer, "boundary_token_index": boundary, "metrics": metrics, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--positive-window", type=int, default=2)
    ap.add_argument("--out", default="experiments/A4_gate_metric/raw_cells.json")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out.parent / "env.json").write_text(json.dumps(result["env"], indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))

if __name__ == "__main__":
    main()
