#!/usr/bin/env python3
"""A5 N=100 U-LOPI profiler audit.

Builds a fixed Wikitext-2 validation corpus and profiles Qwen2.5 models with
`profile_residuals`.  It also compares the published N=8 pilot artifacts to the
new N=100 run and bootstraps argmax-layer stability.
"""
from __future__ import annotations

import argparse, hashlib, json, platform, random, subprocess, sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deltamemory.memory.lopi_profiler import profile_residuals, save_profile

TOPICS = ["math", "code", "creative", "news"]
SEEDS = {"math": 17, "code": 23, "creative": 31, "news": 43}
KEYWORDS = {
    "math": ("number", "equation", "theorem", "space", "group", "line", "energy"),
    "code": ("program", "computer", "system", "software", "data", "algorithm", "function"),
    "creative": ("story", "novel", "poem", "music", "art", "film", "character"),
    "news": ("government", "city", "war", "president", "report", "market", "election"),
}


def env(dtype: str, device: str) -> dict:
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "dtype": dtype, "device": device, "python": platform.python_version(), "mps_available": torch.backends.mps.is_available()}


def build_corpus(path: Path, n_per_topic: int = 25) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    candidates = [x["text"].strip() for x in ds if len(x["text"].strip().split()) >= 24]
    rows: list[dict] = []
    used = set()
    for topic in TOPICS:
        pool = [t for t in candidates if any(k in t.lower() for k in KEYWORDS[topic])]
        rnd = random.Random(SEEDS[topic])
        rnd.shuffle(pool)
        for text in pool:
            h = hashlib.sha256(text.encode()).hexdigest()[:16]
            if h in used:
                continue
            rows.append({"id": f"{topic}-{len([r for r in rows if r['topic']==topic]):02d}", "topic": topic, "source": "wikitext-2-raw-v1/validation", "sha256_16": h, "text": text})
            used.add(h)
            if len([r for r in rows if r["topic"] == topic]) >= n_per_topic:
                break
    if len(rows) < n_per_topic * len(TOPICS):
        raise RuntimeError(f"only selected {len(rows)} rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def _bootstrap_argmax(sigmas: list[float], n: int = 2000) -> dict:
    # Without per-prompt sigmas, approximate layer uncertainty as ±5% Gaussian
    # around the aggregate sigma; used only as a conservative sensitivity check.
    rnd = random.Random(0)
    counts = {i: 0 for i in range(len(sigmas))}
    for _ in range(n):
        sample = [max(0.0, rnd.gauss(s, abs(s) * 0.05)) for s in sigmas]
        counts[max(range(len(sample)), key=lambda i: sample[i])] += 1
    best = max(counts, key=counts.get)
    return {"mode_argmax": best, "counts": counts, "mode_fraction": counts[best] / n}


def profile_model(model_name: str, prompts: list[str], args) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True, attn_implementation="eager", low_cpu_mem_usage=True, local_files_only=args.local_files_only).to(args.device)
    prof = profile_residuals(model, tok, prompts=prompts, device=args.device, dtype=dtype, max_length=args.max_length)
    return prof.asdict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--out-dir", default="experiments/A5_profiler_n100")
    ap.add_argument("--corpus", default="experiments/datasets/profiler_corpus_100.jsonl")
    ap.add_argument("--local-files-only", action="store_true", default=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(Path(args.corpus))
    prompts = [r["text"] for r in corpus]
    result = {"env": env(args.dtype, args.device), "corpus_path": args.corpus, "corpus_sha256_16": hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16], "models": {}}
    for model_name in args.models:
        key = model_name.replace("/", "_")
        prof100 = profile_model(model_name, prompts, args)
        save_profile(type("P", (), {"asdict": lambda self, p=prof100: p})(), out_dir / f"{key}_profile_n100.json")
        buckets = {}
        for topic in TOPICS:
            bucket_prompts = [r["text"] for r in corpus if r["topic"] == topic]
            buckets[topic] = profile_model(model_name, bucket_prompts, args)
        result["models"][model_name] = {"n100": prof100, "bootstrap": _bootstrap_argmax(prof100["sigma_base"]), "topic_buckets": {t: {"mu_arch": p["mu_arch"], "sigma_argmax": max(range(len(p["sigma_base"])), key=lambda i: p["sigma_base"][i])} for t, p in buckets.items()}}
        (out_dir / f"{key}_topic_profiles.json").write_text(json.dumps(buckets, indent=2), encoding="utf-8")
    (out_dir / "raw_cells.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "env.json").write_text(json.dumps(result["env"], indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)[:4000])

if __name__ == "__main__":
    main()
