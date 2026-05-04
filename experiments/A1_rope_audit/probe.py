#!/usr/bin/env python3
"""A1 RoPE audit probe on Qwen2.5-0.5B.

Writes two facts with controlled filler-token gaps, then records per-bank-slot
attention during a read.  The attn-native bank scores bank slots with q_pre @
M_K_pre (no RoPE on bank K); drift across write order/gap should therefore be
near zero except for semantic/filler effects in the captured K itself.
"""
from __future__ import annotations

import argparse, json, platform, subprocess, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltamemory.diagnostics import DiagnosticRecorder
from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank, write_fact


def _env(dtype: str, device: str) -> dict:
    import transformers
    return {
        "torch": torch.__version__, "transformers": transformers.__version__,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dtype": dtype, "device": device, "python": platform.python_version(),
        "mps_available": torch.backends.mps.is_available(),
    }


def _filler_tokens(tok, n: int) -> str:
    if n <= 0:
        return ""
    # Repeated short words keep tokenization deterministic enough; exact count is
    # recorded and trimmed/padded by tokenizer length in the raw output.
    s = " ".join(["therefore"] * (n + 4))
    ids = tok(s, add_special_tokens=False).input_ids[:n]
    return tok.decode(ids, skip_special_tokens=True)


def run(args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager", low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()
    patcher = AttnNativePatcher(model)

    rows = []
    for gap in args.gaps:
        filler = _filler_tokens(tok, gap)
        bank = fresh_bank(model)
        write_fact(patcher, bank, tok, "Fact: The codeword for the blue key is azura.", "blue", "blue key")
        write_fact(patcher, bank, tok, f"{filler} Fact: The codeword for the red key is rubra.", "red", "red key")
        read_prompt = "Question: the blue key codeword and the red key codeword are"
        enc = tok(read_prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(args.device)
        am = enc["attention_mask"].to(args.device)
        with patcher.patched(), patcher.injecting(bank, alpha=args.alpha), DiagnosticRecorder(model, patcher, enabled=True) as rec, torch.no_grad():
            _ = model(input_ids=ids, attention_mask=am, use_cache=False)
        records = [r for r in rec.records if r["signal_name"] == "bank_col_sum"] if hasattr(rec, "records") else []
        # DiagnosticRecorder exposes to_pandas in newer builds; fall back to private records.
        if not records:
            records = [r for r in getattr(rec, "_records", []) if r["signal_name"] == "bank_col_sum"]
        by_slot = {}
        for r in records:
            by_slot.setdefault(int(r["token"]), []).append(float(r["value"]))
        rows.append({
            "gap_requested_tokens": gap,
            "gap_actual_tokens": len(tok(filler, add_special_tokens=False).input_ids),
            "fact_ids": list(bank.fact_ids),
            "bank_col_sum_mean_by_slot": {str(k): sum(v)/len(v) for k, v in by_slot.items()},
            "bank_col_sum_raw": records[:2000],
        })
    return {"env": _env(args.dtype, args.device), "model": args.model, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gaps", type=int, nargs="*", default=[0, 4, 64])
    ap.add_argument("--out", default="experiments/A1_rope_audit/raw_cells.json")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out.parent / "env.json").write_text(json.dumps(result["env"], indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False)[:4000])

if __name__ == "__main__":
    main()
