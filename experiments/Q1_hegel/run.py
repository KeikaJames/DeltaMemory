"""Q.1 — Hegel counterfactual generation runner.

Implements the experiment specified in ``experiments/Q1_hegel/PREREG.md``.

For each (model, prompt, seed, alpha) cell we:
  1. Calibrate a CAA steering vector from (canonical → counterfact) pair.
  2. Install the hook at the given alpha.
  3. Greedy-decode max_new_tokens tokens verbatim.
  4. Remove the hook.
  5. Write the full transcript to ``transcripts/{cell_id}.txt``.
  6. Append a row to ``cells.jsonl``.

Red-line (bit-equality) check: at alpha=0.0 the sha1 of the generated text
must equal the sha1 of a no-injection baseline generation.

Resume-safe: cells whose transcript file already exists are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltamemory.memory.caa_injector import CAAConfig, CAAInjector

PREREG_VERSION = "Q1.v1"

DEFAULT_MODELS = ["gpt2-medium", "Qwen/Qwen2.5-0.5B"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_ALPHAS = [0.0, 0.5, 1.0, 2.0]
DEFAULT_MAX_NEW_TOKENS = 80

PROMPTS_PATH = Path(__file__).parent / "prompts.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sha1_str(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def make_cell_id(model: str, prompt_id: str, seed: int, alpha: float) -> str:
    key = f"{model}|{prompt_id}|{seed}|{alpha}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def load_prompts(n: Optional[int] = None) -> list[dict]:
    rows = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:n] if n is not None else rows


def load_done_cells(cells_path: Path) -> set[str]:
    done: set[str] = set()
    if not cells_path.exists():
        return done
    with cells_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                    cid = row.get("cell_id")
                    if cid:
                        done.add(cid)
                except Exception:
                    pass
    return done


def append_row(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Q1][load] {name} device={device} dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tok, model


def build_fact_line(subject: str, phrase: str, target: str) -> str:
    return f"Fact: {subject} {phrase} {target}."


# ---------------------------------------------------------------------------
# CAA calibration
# ---------------------------------------------------------------------------


def build_caa_injector(
    model: Any,
    tok: Any,
    prompt_row: dict,
    device: str,
) -> CAAInjector:
    """Calibrate a CAA injector for the (counterfact → canonical) pair."""
    cfg = CAAConfig(inject_layer="mu_arch", alpha=1.0, use_lopi_gate=False)
    inj = CAAInjector(model, cfg, tokenizer=tok, device=torch.device(device))
    phrase = prompt_row["relation"]
    pos = build_fact_line(prompt_row["subject"], phrase, prompt_row["counterfact"])
    neg = build_fact_line(prompt_row["subject"], phrase, prompt_row["canonical"])
    inj.calibrate([pos], [neg])
    return inj


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_text(
    model: Any,
    tok: Any,
    prompt: str,
    seed: int,
    max_new_tokens: int,
    device: str,
) -> str:
    """Greedy decode; returns only the newly generated tokens as a string."""
    torch.manual_seed(seed)
    input_ids = tok.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.pad_token_id,
        )
    new_ids = out[0][input_ids.shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Contains checks
# ---------------------------------------------------------------------------


def check_contains(text: str, aliases: list[str]) -> bool:
    tl = text.lower()
    return any(alias.lower() in tl for alias in aliases)


def check_contains_canonical(text: str, canonical: str) -> bool:
    return canonical.lower() in text.lower()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    import sys as _sys
    from tools.env_writer import write_env_json, sha1_of

    out_dir = Path(args.out)
    transcripts_dir = out_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    cells_path = out_dir / "cells.jsonl"
    dataset_sha1 = sha1_of(PROMPTS_PATH)

    # Write env.json at start of run
    write_env_json(
        out_dir,
        prereg_version=PREREG_VERSION,
        dataset_sha1=dataset_sha1,
        device=args.device,
        dtype=args.dtype,
        cli_argv=_sys.argv,
        extra={"seeds": args.seeds, "alphas": args.alphas, "models": args.models},
    )

    done_cells = load_done_cells(cells_path)
    prompts = load_prompts(args.n_prompts)

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map.get(args.dtype, torch.float32)

    for model_name in args.models:
        print(f"\n[Q1] === Model: {model_name} ===", flush=True)
        try:
            tok, model = load_model(model_name, args.device, dtype)
        except Exception as exc:
            print(f"[Q1][ERROR] model load failed: {exc}", file=sys.stderr)
            for prompt_row in prompts:
                for seed in args.seeds:
                    for alpha in args.alphas:
                        cid = make_cell_id(model_name, prompt_row["id"], seed, alpha)
                        row = {
                            "cell_id": cid,
                            "model": model_name,
                            "prompt_id": prompt_row["id"],
                            "seed": seed,
                            "alpha": alpha,
                            "status": "model_load_failed",
                            "dataset_sha1": dataset_sha1,
                        }
                        append_row(cells_path, row)
            continue

        for prompt_row in prompts:
            pid = prompt_row["id"]
            prompt_text = prompt_row["prompt"]
            aliases = prompt_row["aliases"]
            canonical = prompt_row["canonical"]

            print(f"[Q1]   prompt={pid}", flush=True)

            # Calibrate once per (model, prompt)
            try:
                inj = build_caa_injector(model, tok, prompt_row, args.device)
            except Exception as exc:
                print(f"[Q1][ERROR] CAA calibration failed for {pid}: {exc}", file=sys.stderr)
                inj = None

            for seed in args.seeds:
                # Generate no-injection baseline once per (model, prompt, seed)
                # needed for red-line check at alpha=0
                try:
                    baseline_text = generate_text(model, tok, prompt_text, seed, args.max_new_tokens, args.device)
                    baseline_sha1 = sha1_str(baseline_text)
                except Exception as exc:
                    print(f"[Q1][ERROR] baseline generation failed: {exc}", file=sys.stderr)
                    baseline_sha1 = None
                    baseline_text = None

                for alpha in args.alphas:
                    cid = make_cell_id(model_name, pid, seed, alpha)

                    # Resume: skip if transcript already exists
                    transcript_path = transcripts_dir / f"{cid}.txt"
                    if cid in done_cells or transcript_path.exists():
                        print(f"[Q1]     skip (resume) cell={cid}", flush=True)
                        continue

                    print(f"[Q1]     seed={seed} alpha={alpha} cell={cid}", flush=True)

                    if inj is None:
                        row = {
                            "cell_id": cid,
                            "model": model_name,
                            "prompt_id": pid,
                            "seed": seed,
                            "alpha": alpha,
                            "status": "calibration_failed",
                            "dataset_sha1": dataset_sha1,
                        }
                        append_row(cells_path, row)
                        continue

                    try:
                        inj.config.alpha = float(alpha)
                        with inj:
                            gen_text = generate_text(
                                model, tok, prompt_text, seed,
                                args.max_new_tokens, args.device
                            )

                        gen_sha1 = sha1_str(gen_text)

                        # Red-line check: alpha=0 must be bit-identical to baseline
                        redline_ok: Optional[bool] = None
                        if abs(alpha) < 1e-9:
                            if baseline_sha1 is not None:
                                redline_ok = (gen_sha1 == baseline_sha1)
                                if not redline_ok:
                                    print(
                                        f"[REDLINE VIOLATION] cell={cid} "
                                        f"model={model_name} prompt={pid} seed={seed}",
                                        file=sys.stderr,
                                    )
                            else:
                                redline_ok = None

                        # Write verbatim transcript
                        transcript_path.write_text(gen_text, encoding="utf-8")

                        contains_cf = check_contains(gen_text, aliases)
                        contains_can = check_contains_canonical(gen_text, canonical)

                        row: dict = {
                            "cell_id": cid,
                            "model": model_name,
                            "prompt_id": pid,
                            "seed": seed,
                            "alpha": alpha,
                            "contains_counterfact": contains_cf,
                            "contains_canonical": contains_can,
                            "full_text_sha1": gen_sha1,
                            "transcript_path": str(transcript_path.relative_to(out_dir)),
                            "status": "ok",
                            "dataset_sha1": dataset_sha1,
                        }
                        if redline_ok is not None:
                            row["redline_ok"] = redline_ok

                        append_row(cells_path, row)
                        done_cells.add(cid)

                    except Exception as exc:
                        print(
                            f"[Q1][ERROR] cell={cid}: {exc}\n{traceback.format_exc()}",
                            file=sys.stderr,
                        )
                        row = {
                            "cell_id": cid,
                            "model": model_name,
                            "prompt_id": pid,
                            "seed": seed,
                            "alpha": alpha,
                            "status": f"error:{type(exc).__name__}",
                            "dataset_sha1": dataset_sha1,
                        }
                        append_row(cells_path, row)

    print(f"\n[Q1] Done. cells.jsonl: {cells_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q.1 Hegel CAA generation runner")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "float32", "bfloat16"])
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    p.add_argument("--n-prompts", type=int, default=None, help="Use first N prompts (default: all 5)")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: gpt2-medium, 1 prompt, 1 seed, alpha in {0.0, 1.0}",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    if args.smoke:
        args.models = ["gpt2-medium"]
        args.seeds = [0]
        args.alphas = [0.0, 1.0]
        args.n_prompts = 1
        if args.max_new_tokens == DEFAULT_MAX_NEW_TOKENS:
            args.max_new_tokens = 80

    run(args)


if __name__ == "__main__":
    main()
