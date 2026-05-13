"""W.7 long-context degradation runner.

Implements the experiment specified in
``experiments/W7_longctx/PREREG.md``.  Per cell we measure:

    nll_target       — mean NLL over the last 8 tokens of the gold suffix.
    top1_match_frac  — fraction of suffix tokens where argmax(M_winner)
                       equals argmax(M_none).
    kv_cache_mb      — peak KV-cache memory (audit only).

Cell key
--------
``cell_id = sha1(f"{model}|{method}|{alpha}|{seed}|{length}|{prompt_id}")``

Red-line
--------
For every alpha=0 cell with method != none we verify
``|nll_target(M_winner) - nll_target(M_none)| < 1e-4`` ; otherwise the row is
flagged ``redline_violation=true``.  The runner does NOT abort on a single
violation — the row is persisted and aggregation flags it.

Failed cells
------------
Any cell that raises during evaluation is still written with
``status="failed"`` plus the exception trace, so a partial run never loses
provenance.  Successful cells get ``status="ok"``.

Smoke
-----
``--smoke`` runs gpt2-medium (or fallback) at 1 length × 1 alpha × 1 seed ×
5 prompts × 2 methods = 10 cells, exercising every code path including the
alpha=0 bit-equality check.

Full grid (PREREG section 3)
----------------------------
``5 models x 6 lengths x 7 alphas x 3 seeds x 30 prompts x 2 methods``.
The runner is capable of this via explicit CLI args, but the parent task
explicitly forbids launching it in this session (it is GB10-bound; 2048-token
KV cache on 5 models is hours of forward time).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltamemory.memory.attn_native_bank import (
    AttnNativeBank,
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.caa_injector import CAAConfig, CAAInjector
from deltamemory.memory.lopi import LOPIConfig, LOPIState


# ---------------------------------------------------------------------------
# Constants

PREREG_VERSION = "W.7.v1"

MODELS = [
    "gpt2-medium",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "google/gemma-3-270m",
    "google/gemma-3-1b-it",
]

LENGTHS = [64, 128, 256, 512, 1024, 2048]
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = [0, 1, 2]
SUFFIX_TOKENS = 8

REDLINE_TOL = 1e-4

DATA_ROOT = ROOT / "experiments" / "datasets"
PROMPTS_PATH = DATA_ROOT / "gold_30prompts.jsonl"
CAA_PAIRS_PATH = DATA_ROOT / "caa_calibration_pairs.jsonl"

WRITE_PROMPT = "Fact: The Sun is a star at the centre of the Solar System."
WRITE_FACT_ID = "neutral_anchor"
WRITE_ADDRESS = "the Sun"


# ---------------------------------------------------------------------------
# Utility


def cell_id(model: str, method: str, alpha: float, seed: int,
            length: int, prompt_id: str) -> str:
    key = f"{model}|{method}|{alpha}|{seed}|{length}|{prompt_id}"
    return hashlib.sha1(key.encode()).hexdigest()


def append_cell(path: Path, row: dict) -> None:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "at") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def load_done_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    open_fn = gzip.open if str(path).endswith(".gz") else open
    try:
        with open_fn(path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cid = row.get("cell_id")
                if cid:
                    done.add(cid)
    except Exception:
        pass
    return done


def env_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def is_rope_model(name: str) -> bool:
    return name.startswith("Qwen/") or name.startswith("google/gemma") \
        or "llama" in name.lower()


def is_gemma(name: str) -> bool:
    return name.startswith("google/gemma")


# ---------------------------------------------------------------------------
# W.4 winner resolution (mirrors W.6 logic)


def resolve_method_winner(force_smoke: bool = False) -> tuple[str, str]:
    if force_smoke:
        return "caa", "smoke_assumption"
    rep = ROOT / "experiments" / "W4_caa_baseline" / "REPORT.md"
    if rep.exists():
        try:
            text = rep.read_text(errors="ignore")
            if "H1 PASS" in text and "caa" in text:
                return "caa", "w4_h1_passed"
        except Exception:
            pass
    return "caa", "w4_default"


# ---------------------------------------------------------------------------
# Prompt prep


def load_prompts() -> list[dict]:
    rows: list[dict] = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_caa_pairs() -> tuple[list[str], list[str]]:
    pos: list[str] = []
    neu: list[str] = []
    with open(CAA_PAIRS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pos.append(row["positive"])
            neu.append(row["neutral"])
    return pos, neu


def build_length_input(tok, prompt_text: str, target_length: int,
                       all_texts: list[str], seed: int) -> torch.Tensor:
    """Tokenize prompt_text, then either truncate or extend deterministically
    by concatenating other prompts (seeded shuffle) until target_length is
    reached.  Returns a 1D LongTensor of exactly ``target_length`` ids
    (terminating on the original prompt_text suffix so the last
    ``SUFFIX_TOKENS`` are gold).
    """
    suffix_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    if len(suffix_ids) >= target_length:
        ids = suffix_ids[-target_length:]
        return torch.tensor(ids, dtype=torch.long)

    # Need prefix.  Build a deterministic shuffle of available prefixes.
    rng = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(len(all_texts), generator=rng).tolist()
    prefix_pool: list[int] = []
    for idx in order:
        if all_texts[idx] == prompt_text:
            continue
        prefix_pool.extend(tok(all_texts[idx],
                               add_special_tokens=False)["input_ids"])
        if len(prefix_pool) + len(suffix_ids) >= target_length:
            break

    # If still short, repeat prefix pool deterministically.
    while len(prefix_pool) + len(suffix_ids) < target_length:
        prefix_pool.extend(prefix_pool[: target_length])

    needed = target_length - len(suffix_ids)
    ids = prefix_pool[:needed] + suffix_ids
    assert len(ids) == target_length, (len(ids), target_length)
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Forward passes


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)


def _suffix_metrics(logits_inj: torch.Tensor, logits_base: torch.Tensor,
                    ids: torch.Tensor, suffix_n: int = SUFFIX_TOKENS
                    ) -> tuple[float, float]:
    """Return (nll_target, top1_match_frac) over the last ``suffix_n`` tokens.

    logits_*: [1, T, V].  ids: [1, T].  Targets are ids[:, -suffix_n:].
    Predictions for token t come from logits[:, t-1].
    """
    targets = ids[0, -suffix_n:]
    pred_logits_inj = logits_inj[0, -suffix_n - 1:-1]   # [suffix_n, V]
    pred_logits_base = logits_base[0, -suffix_n - 1:-1]
    logp_inj = F.log_softmax(pred_logits_inj.float(), dim=-1)
    nll_target = float(
        -logp_inj.gather(1, targets.unsqueeze(-1)).squeeze(-1).mean().item()
    )
    top1_inj = pred_logits_inj.argmax(dim=-1)
    top1_base = pred_logits_base.argmax(dim=-1)
    top1_match_frac = float((top1_inj == top1_base).float().mean().item())
    return nll_target, top1_match_frac


def forward_baseline(model, ids_full: torch.Tensor, device: str
                     ) -> torch.Tensor:
    ids = ids_full.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    return out.logits  # [1, T, V]


def forward_with_method(model, tok, ids_full: torch.Tensor, method: str,
                        alpha: float, seed: int, device: str,
                        bank=None, patcher=None, caa_inj=None) -> torch.Tensor:
    _set_seeds(seed)
    ids = ids_full.unsqueeze(0).to(device)

    if method == "none":
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        return out.logits

    if method == "lopi_default":
        assert bank is not None and patcher is not None
        bank.lopi_cfg = LOPIConfig(
            enabled=True, orthogonal=True, gaussian=True, derivative=True,
        )
        bank.lopi_state = LOPIState(num_layers=bank.num_layers)
        bank.lopi_state.reset()
        bank.mhc_shield = False
        with patcher.patched(), patcher.injecting(bank, alpha=alpha), \
                torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        return out.logits

    if method == "caa":
        assert caa_inj is not None
        caa_inj.config.alpha = float(alpha)
        with caa_inj, torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        return out.logits

    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# Model loading


def load_model(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {model_name}  device={device}  dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tok, model


# ---------------------------------------------------------------------------
# Main


def write_env(env_path: Path, args: argparse.Namespace, models: list[str],
              methods: list[str], lengths: list[int], alphas: list[float],
              seeds: list[int], n_prompts: int, M_winner: str,
              winner_source: str) -> None:
    try:
        import transformers as _tf
        transformers_ver = _tf.__version__
    except Exception:
        transformers_ver = None
    total_cells = (
        len(models) * len(methods) * len(lengths)
        * len(alphas) * len(seeds) * n_prompts
    )
    env = {
        "prereg_version": PREREG_VERSION,
        "method_winner": M_winner,
        "method_winner_source": winner_source,
        "models": models,
        "methods": methods,
        "lengths": lengths,
        "alphas": alphas,
        "seeds": seeds,
        "n_prompts": n_prompts,
        "smoke": bool(args.smoke),
        "torch": torch.__version__,
        "transformers": transformers_ver,
        "device": args.device,
        "dtype": args.dtype,
        "git_commit": env_commit(),
        "total_cells_planned": total_cells,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    env_path.parent.mkdir(parents=True, exist_ok=True)
    with open(env_path, "w") as f:
        json.dump(env, f, indent=2, sort_keys=True)
    print(f"[W7] env -> {env_path}  total_cells_planned={total_cells}",
          flush=True)


def run_grid(args: argparse.Namespace) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    M_winner, source = resolve_method_winner(force_smoke=args.smoke)

    if args.smoke:
        models = [args.models[0]] if args.models else ["gpt2-medium"]
        methods = ["none", M_winner]
        lengths = [64]
        alphas = [0.0, 1.0]
        seeds = [0]
        n_prompts = 5
    else:
        models = list(args.models)
        methods = ["none", M_winner]
        lengths = list(args.lengths)
        alphas = list(args.alphas)
        seeds = list(args.seeds)
        n_prompts = args.n_prompts

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = args.device

    all_prompts = load_prompts()[:n_prompts]
    all_texts = [p["text"] for p in all_prompts]
    pos_texts, neu_texts = load_caa_pairs()

    env_path = out_path.parent / "env.json"
    write_env(env_path, args, models, methods, lengths, alphas, seeds,
              len(all_prompts), M_winner, source)

    done = load_done_keys(out_path)
    print(f"[W7] resumed_done={len(done)}", flush=True)

    total_target = (len(models) * len(methods) * len(lengths)
                    * len(alphas) * len(seeds) * len(all_prompts))
    passes = 0
    t0 = time.time()

    for model_name in models:
        try:
            tok, model = load_model(model_name, device, dtype)
        except Exception as exc:
            print(f"[W7][skip-model] {model_name}: {exc!r}", flush=True)
            traceback.print_exc()
            continue

        is_gpt2 = model_name.startswith("gpt2")
        is_rope = is_rope_model(model_name)
        is_gem = is_gemma(model_name)

        # Build bank if needed for lopi_default winner.
        bank = None
        patcher = None
        if "lopi_default" in methods and is_rope:
            try:
                patcher = AttnNativePatcher(model)
                bank = fresh_bank(model)
                bank.mhc_shield = False
                if is_gem:
                    bank.value_scale_mode = "none"
                else:
                    bank.value_scale_mode = "auto_rms_cap"
                    bank.value_target_rms = 0.5
                write_fact(patcher, bank, tok,
                           write_prompt=WRITE_PROMPT,
                           fact_id=WRITE_FACT_ID,
                           address=WRITE_ADDRESS)
                try:
                    bank.attach_lopi_profile(model, tok)
                except Exception as exc:
                    print(f"[W7] attach_lopi_profile: {exc!r}", flush=True)
            except Exception as exc:
                print(f"[W7] bank setup failed: {exc!r}", flush=True)
                bank = None
                patcher = None

        # Build CAA injector if needed.
        caa_inj = None
        if "caa" in methods:
            try:
                caa_cfg = CAAConfig(
                    inject_layer="mu_arch",
                    alpha=1.0,
                    use_lopi_gate=False,
                )
                caa_inj = CAAInjector(model, caa_cfg, tokenizer=tok)
                caa_inj.calibrate(pos_texts, neu_texts)
                print(f"[W7] CAA calibrated layer={caa_inj._inject_layer}",
                      flush=True)
            except Exception as exc:
                print(f"[W7] CAA setup failed: {exc!r}", flush=True)
                traceback.print_exc()
                caa_inj = None

        for length in lengths:
            for seed in seeds:
                # Build per-(seed, length, prompt) input ids once.
                ids_cache: dict[str, torch.Tensor] = {}
                for prow in all_prompts:
                    try:
                        ids_cache[prow["id"]] = build_length_input(
                            tok, prow["text"], length, all_texts, seed)
                    except Exception as exc:
                        print(f"[W7] build_length_input failed "
                              f"{prow['id']} L={length}: {exc!r}", flush=True)

                # Cache base logits per prompt for top1 comparison + redline.
                base_cache: dict[str, torch.Tensor] = {}

                for method in methods:
                    if method == "lopi_default" and (is_gpt2 or bank is None):
                        # Architecture-incompatible.  Sentinel.
                        cid = cell_id(model_name, method, -1.0, -1, length,
                                      "__unsupported__")
                        if cid not in done:
                            row = {
                                "cell_id": cid, "model": model_name,
                                "method": method, "alpha": -1.0, "seed": -1,
                                "length": length,
                                "prompt_id": "__unsupported__",
                                "nll_target": None,
                                "top1_match_frac": None,
                                "kv_cache_mb": None,
                                "drift": None,
                                "method_unsupported": True,
                                "redline_violation": False,
                                "status": "unsupported",
                            }
                            append_cell(out_path, row)
                            done.add(cid)
                        continue
                    if method == "caa" and caa_inj is None:
                        cid = cell_id(model_name, method, -1.0, -1, length,
                                      "__unsupported__")
                        if cid not in done:
                            row = {
                                "cell_id": cid, "model": model_name,
                                "method": method, "alpha": -1.0, "seed": -1,
                                "length": length,
                                "prompt_id": "__unsupported__",
                                "nll_target": None,
                                "top1_match_frac": None,
                                "kv_cache_mb": None,
                                "drift": None,
                                "method_unsupported": True,
                                "redline_violation": False,
                                "status": "unsupported",
                            }
                            append_cell(out_path, row)
                            done.add(cid)
                        continue

                    for alpha in alphas:
                        for prow in all_prompts:
                            cid = cell_id(model_name, method, alpha, seed,
                                          length, prow["id"])
                            if cid in done:
                                passes += 1
                                continue
                            if prow["id"] not in ids_cache:
                                continue
                            ids_full = ids_cache[prow["id"]]
                            try:
                                # Compute base logits once per (prompt, seed).
                                if prow["id"] not in base_cache:
                                    base_cache[prow["id"]] = forward_baseline(
                                        model, ids_full, device)
                                base_logits = base_cache[prow["id"]]

                                if method == "none":
                                    inj_logits = base_logits
                                else:
                                    inj_logits = forward_with_method(
                                        model, tok, ids_full, method, alpha,
                                        seed, device, bank=bank,
                                        patcher=patcher, caa_inj=caa_inj,
                                    )
                                nll_t, top1f = _suffix_metrics(
                                    inj_logits, base_logits,
                                    ids_full.unsqueeze(0).to(device))

                                # Drift = inj - base (computed at suffix).
                                if method == "none":
                                    drift = 0.0
                                else:
                                    base_nll, _ = _suffix_metrics(
                                        base_logits, base_logits,
                                        ids_full.unsqueeze(0).to(device))
                                    drift = float(nll_t - base_nll)

                                redline = (
                                    method != "none"
                                    and float(alpha) == 0.0
                                    and abs(drift) >= REDLINE_TOL
                                )
                                if redline:
                                    print(f"[REDLINE] {model_name} {method} "
                                          f"L={length} s={seed} "
                                          f"prompt={prow['id']} "
                                          f"|drift|={abs(drift):.3e}",
                                          file=sys.stderr, flush=True)

                                # KV cache audit (best-effort).
                                kv_mb = None
                                if device == "cuda":
                                    try:
                                        kv_mb = float(
                                            torch.cuda.max_memory_allocated()
                                            / (1024 ** 2))
                                    except Exception:
                                        kv_mb = None

                                row = {
                                    "cell_id": cid,
                                    "model": model_name,
                                    "method": method,
                                    "alpha": float(alpha),
                                    "seed": int(seed),
                                    "length": int(length),
                                    "prompt_id": prow["id"],
                                    "nll_target": float(nll_t),
                                    "top1_match_frac": float(top1f),
                                    "kv_cache_mb": kv_mb,
                                    "drift": float(drift),
                                    "method_unsupported": False,
                                    "redline_violation": bool(redline),
                                    "status": "ok",
                                }
                            except Exception as exc:
                                tb = traceback.format_exc()
                                print(f"[W7][ERROR] {model_name} {method} "
                                      f"a={alpha} s={seed} L={length} "
                                      f"p={prow['id']}: {exc!r}",
                                      file=sys.stderr, flush=True)
                                traceback.print_exc()
                                row = {
                                    "cell_id": cid,
                                    "model": model_name,
                                    "method": method,
                                    "alpha": float(alpha),
                                    "seed": int(seed),
                                    "length": int(length),
                                    "prompt_id": prow["id"],
                                    "nll_target": None,
                                    "top1_match_frac": None,
                                    "kv_cache_mb": None,
                                    "drift": None,
                                    "method_unsupported": False,
                                    "redline_violation": False,
                                    "status": "failed",
                                    "exc": repr(exc),
                                    "traceback": tb,
                                }

                            append_cell(out_path, row)
                            done.add(cid)
                            passes += 1

                        elapsed = time.time() - t0
                        eta = (elapsed / max(passes, 1)) \
                            * max(total_target - passes, 0)
                        print(f"[{passes:>5}/{total_target}] {model_name} "
                              f"method={method} L={length} alpha={alpha} "
                              f"seed={seed} elapsed={elapsed:.1f}s "
                              f"eta={eta/60:.1f}m", flush=True)

        del model, tok
        if device == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass
        elif device == "cuda":
            try: torch.cuda.empty_cache()
            except Exception: pass

    print(f"[W7] DONE  passes={passes}/{total_target}  "
          f"elapsed={(time.time()-t0)/60:.2f}m", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="W.7 long-context runner")
    ap.add_argument("--out",
                    default=str(ROOT / "experiments" / "W7_longctx"
                                / "cells.jsonl"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke: 1 model x [none, winner] x L=64 x "
                         "alpha in {0,1} x seed=0 x 5 prompts -> 20 cells. "
                         "out -> cells_smoke.jsonl.")
    args = ap.parse_args()

    if args.smoke and args.out.endswith("cells.jsonl"):
        args.out = args.out.replace("cells.jsonl", "cells_smoke.jsonl")

    run_grid(args)


if __name__ == "__main__":
    main()
