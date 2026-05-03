"""Phase M — v3.1 multi-model benchmark.

Runs B0 (no-memory), B1 (prompt-insertion), and B6 (v3.1 K-projector)
on any HF model with ArchAdapter support.  Outputs per-fact recall@1 JSON.

The benchmark split is val2_v31.jsonl (sha-locked, 41 held-out facts
from 15 LAMA relations × 305 facts × 10 paraphrases).

Usage (single model, Mac MPS):
    .venv-mac/bin/python scripts/run_v31_benchmark.py \
        --model google/gemma-4-E2B --device mps --dtype bfloat16 \
        --split eval/splits_v31/val2_v31.jsonl \
        --projector reports/cleanroom/stage15_kproj_v31/k_projector_gemma4.pt \
        --seeds 0 1 2 --out reports/cleanroom/v31_bench

Usage (multi-model, GB10):
    for m in google/gemma-4-E2B Qwen/Qwen3-4B-Instruct-2507 \
             deepseek-ai/DeepSeek-R1-Distill-Qwen-32B THUDM/GLM-4-9B-0414; do
        python scripts/run_v31_benchmark.py --model "$m" --device cuda \
            --split eval/splits_v31/val2_v31.jsonl --seeds 0 1 2 \
            --out reports/cleanroom/v31_bench
    done
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.run_stage14_test_eval import _load_test, _recall_with_bank
from deltamemory.memory.attn_native_bank import AttnNativePatcher
from deltamemory.memory.k_projector import KProjectorBank


def _prompt_insertion_recall(model, tok, facts, seed: int) -> list[float]:
    """Baseline B1: prepend value to prompt (oracle retrieval)."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    per_fact = []
    for f in facts:
        addr = f.address
        value = f.value
        prompt = f"Fact: {addr} is {value}.\nQ: {addr}?\nA:"
        enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=am, use_cache=True)
        last = am.sum(dim=1).item() - 1
        logits = out.logits[0, last].float()
        target_id = tok(" " + value, add_special_tokens=False)["input_ids"][0]
        # Check if target is top-1
        top1 = logits.argmax().item()
        per_fact.append(1.0 if top1 == target_id else 0.0)
    return per_fact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--split", default="eval/splits_v31/val2_v31.jsonl")
    ap.add_argument("--projector", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--baselines", nargs="+",
                    default=["B0", "B1", "B6"],
                    choices=["B0", "B1", "B6"])
    ap.add_argument("--out", default="reports/cleanroom/v31_bench")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    short = args.model.replace("/", "_")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    facts = _load_test(Path(args.split))
    print(f"[bench] model={args.model}  device={args.device}  N={len(facts)} facts", flush=True)

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    print(f"[bench] loaded in {time.time()-t_load:.1f}s", flush=True)

    patcher = AttnNativePatcher(model)
    kproj = None
    if args.projector and "B6" in args.baselines:
        kproj = KProjectorBank.load(args.projector)
        print(f"[bench] loaded K-projector from {args.projector}", flush=True)

    results: dict[str, list] = {}

    for seed in args.seeds:
        for baseline in args.baselines:
            label = f"{baseline}_seed{seed}"
            t0 = time.time()

            if baseline == "B0":
                r = _recall_with_bank(patcher, tok, facts, policy="period",
                                      k_projector=None, tau=1.0, seed=seed)
            elif baseline == "B1":
                r = _prompt_insertion_recall(model, tok, facts, seed=seed)
            elif baseline == "B6":
                if kproj is None:
                    continue
                r = _recall_with_bank(patcher, tok, facts, policy="period",
                                      k_projector=kproj, tau=1.0, seed=seed)
            else:
                continue

            mean_r = sum(r) / max(len(r), 1)
            results[label] = dict(recall_per_fact=r, recall_mean=round(mean_r, 4))
            print(f"  [{label}] recall@1={mean_r:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    # Save
    out_path = out_dir / f"{short}.json"
    summary = dict(
        model=args.model, split=args.split,
        n_facts=len(facts), seeds=args.seeds,
        baselines=args.baselines,
        results=results,
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[bench] wrote {out_path}", flush=True)

    # Quick table
    print(f"\n[bench] === {short} ===")
    for baseline in args.baselines:
        vals = [results[f"{baseline}_seed{s}"]["recall_mean"]
                for s in args.seeds if f"{baseline}_seed{s}" in results]
        if vals:
            mean = sum(vals) / len(vals)
            print(f"  {baseline}: recall@1 = {mean:.4f}  (n={len(vals)} seeds)")


if __name__ == "__main__":
    main()
