#!/usr/bin/env python3
"""S-7: U-LOPI cross-arch MPS pilot — paired static vs auto drift sweep.

Empirical validation that the Phase-S U-LOPI auto-calibration profile
matches or beats the v3.4 hard-coded ``norm_base=10.0`` static path on a
non-Gemma architecture small enough to fit on Apple MPS bf16.

For each (model × alpha × seed) cell we run BOTH:

* ``profile_mode='static'`` — legacy, ``norm_base=10.0``.
* ``profile_mode='auto'``   — Z-score profile attached via
  ``AttnNativeBank.attach_lopi_profile(model, tokenizer)``.

Held-out eval = mean per-token NLL over a small neutral set forwarded
through the bank-injected model with the given alpha.  Each cell writes
exactly one fact to a fresh bank so the write itself is identical
between static/auto modes; only the depth signal differs.

Red lines (asserted at runtime):
  * No nn.Parameter created, no gradient (``torch.set_grad_enabled(False)``).
  * alpha=0 ⇒ static and auto must be bit-equal (max_abs_diff < 1e-6) on
    the eval logits, since LOPI is multiplicative on the bank readout.
  * If a model fails to load on MPS we log and try the next.
  * If all models fail, the report documents the blocker.

Usage
-----
    python scripts/run_ulopi_xarch_smoke.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --device mps --dtype bfloat16 \\
        --alpha 0,2,5 --seeds 0,1,2 --n-prompts 8 \\
        --out reports/cleanroom/ulopi_xarch/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deltamemory import (
    AttnNativeBank,
    AttnNativePatcher,
    LOPIConfig,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.lopi import LOPIState


# ---------------------------------------------------------------------------
# Held-out neutral prompts (8 entries) — Wikipedia-style declaratives, no
# overlap with the U-LOPI default profile corpus and no overlap with the
# write-fact subject (the Sun).
EVAL_PROMPTS: tuple[str, ...] = (
    "The boiling point of water at sea level is one hundred degrees Celsius.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "The Pacific Ocean covers approximately one third of the Earth's surface area.",
    "DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
    "The speed of light in a vacuum is approximately three hundred thousand kilometers per second.",
    "Mount Everest is the tallest mountain above sea level, located in the Himalayan range.",
    "The Industrial Revolution began in Britain during the late eighteenth century.",
    "Quantum mechanics describes the behavior of matter and energy at atomic scales.",
)

WRITE_PROMPT = "Fact: The Sun is a star at the centre of the Solar System."
WRITE_FACT_ID = "neutral_anchor"
WRITE_ADDRESS = "the Sun"


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _try_load(name: str, device: str, dtype: torch.dtype):
    """Load (tokenizer, model) on device/dtype.  Returns (tok, model) or None on failure."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {name}  device={device}  dtype={dtype}", flush=True)
    try:
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
        # Smoke forward — surfaces missing-adapter / OOM early.
        with torch.no_grad():
            ids = tok("hello", return_tensors="pt").input_ids.to(device)
            model(input_ids=ids, use_cache=False)
        return tok, model
    except Exception as exc:  # pragma: no cover - environmental
        print(f"[load-fail] {name}: {exc!r}", flush=True)
        traceback.print_exc()
        return None


@torch.no_grad()
def _eval_logits_and_nll(patcher: AttnNativePatcher,
                         bank: AttnNativeBank,
                         tok,
                         prompts,
                         alpha: float):
    """Mean per-token NLL + final-token logits for each eval prompt under bank+alpha."""
    device = next(patcher.model.parameters()).device
    nlls = []
    last_logits = []
    with patcher.patched(), patcher.injecting(bank, alpha=alpha):
        for p in prompts:
            enc = tok(p, return_tensors="pt", add_special_tokens=True)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            out = patcher.model(input_ids=ids, attention_mask=am, use_cache=False)
            logits = out.logits[0]
            targets = ids[0, 1:]
            logp = F.log_softmax(logits[:-1].float(), dim=-1)
            nll = -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1)
            nlls.append(float(nll.mean().item()))
            last = int(am.sum(dim=1).item() - 1)
            last_logits.append(logits[last].float().detach().cpu())
    return nlls, last_logits


@torch.no_grad()
def _baseline_nll(model, tok, prompts):
    device = next(model.parameters()).device
    nlls = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        am = enc["attention_mask"].to(device)
        out = model(input_ids=ids, attention_mask=am, use_cache=False)
        logits = out.logits[0]
        targets = ids[0, 1:]
        logp = F.log_softmax(logits[:-1].float(), dim=-1)
        nll = -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        nlls.append(float(nll.mean().item()))
    return nlls


def _configure_bank(bank: AttnNativeBank, mode: str, model, tok):
    """Set bank.lopi_cfg / lopi_state for the requested mode and (auto) attach profile."""
    bank.lopi_cfg = LOPIConfig(
        enabled=True,
        orthogonal=False,
        gaussian=True,
        derivative=True,
        profile_mode=mode,
    )
    bank.lopi_state = LOPIState(num_layers=bank.num_layers)
    if mode == "auto":
        bank.attach_lopi_profile(model, tok)


def run_cell(model_name: str, model, tok, alpha: float, seed: int, prompts):
    """Run paired (static, auto) for one (alpha, seed) cell.  Returns dict."""
    out = {"alpha": float(alpha), "seed": int(seed)}
    static_nll_per_cell = {}
    last_logits = {}
    for mode in ("static", "auto"):
        torch.manual_seed(seed)
        patcher = AttnNativePatcher(model)
        bank = fresh_bank(model)
        _configure_bank(bank, mode, model, tok)
        write_fact(patcher, bank, tok,
                   write_prompt=WRITE_PROMPT,
                   fact_id=WRITE_FACT_ID,
                   address=WRITE_ADDRESS)
        if mode == "auto":
            prof = bank.lopi_state.profile
            out["mu_arch"] = int(prof.mu_arch)
            out["eta_sigma"] = float(prof.eta_sigma)
            out["mu_base"] = [float(x) for x in prof.mu_base]
            out["sigma_base"] = [float(x) for x in prof.sigma_base]
            out["profile_corpus_sha"] = prof.profile_corpus_sha
        nlls, ll = _eval_logits_and_nll(patcher, bank, tok, prompts, alpha)
        static_nll_per_cell[mode] = nlls
        last_logits[mode] = ll
    out["static_nll_list"] = static_nll_per_cell["static"]
    out["auto_nll_list"] = static_nll_per_cell["auto"]
    out["static_nll"] = float(sum(out["static_nll_list"]) / len(out["static_nll_list"]))
    out["auto_nll"] = float(sum(out["auto_nll_list"]) / len(out["auto_nll_list"]))

    # alpha=0 bit-equal sanity (red line).
    if abs(alpha) < 1e-12:
        max_diff = 0.0
        for s, a in zip(last_logits["static"], last_logits["auto"]):
            d = float((s - a).abs().max().item())
            if d > max_diff:
                max_diff = d
        out["alpha0_max_abs_diff"] = max_diff
        # Loose threshold: bf16 numerics may produce small noise from the
        # different layer-Gaussian scalar paths even though the bank
        # contribution is multiplied by alpha=0.  Keep <1e-6 per spec.
        assert max_diff < 1e-6, (
            f"alpha=0 sanity failed: static vs auto max_abs_diff={max_diff:.3e}"
        )
    return out


def parse_csv_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_csv_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def aggregate(cells, model_name: str):
    """Group by alpha, mean (static, auto, drift_static, drift_auto)."""
    by_alpha: dict[float, dict] = {}
    for c in cells:
        a = c["alpha"]
        by_alpha.setdefault(a, {"static": [], "auto": [],
                                "drift_static": [], "drift_auto": []})
        by_alpha[a]["static"].append(c["static_nll"])
        by_alpha[a]["auto"].append(c["auto_nll"])
        by_alpha[a]["drift_static"].append(c["static_nll"] - c["base_nll"])
        by_alpha[a]["drift_auto"].append(c["auto_nll"] - c["base_nll"])
    rows = []
    for a in sorted(by_alpha.keys()):
        r = by_alpha[a]
        rows.append({
            "alpha": a,
            "n": len(r["static"]),
            "static_nll_mean": sum(r["static"]) / len(r["static"]),
            "auto_nll_mean": sum(r["auto"]) / len(r["auto"]),
            "drift_static_mean": sum(r["drift_static"]) / len(r["drift_static"]),
            "drift_auto_mean": sum(r["drift_auto"]) / len(r["drift_auto"]),
        })
    # Verdict on mean over alpha>0 cells.
    pos = [r for r in rows if r["alpha"] > 0]
    if pos:
        ds = sum(r["drift_static_mean"] for r in pos) / len(pos)
        da = sum(r["drift_auto_mean"] for r in pos) / len(pos)
        if abs(ds - da) < 0.05:
            verdict = "NO_DIFFERENCE"
        elif da < ds:
            verdict = "AUTO<STATIC"
        else:
            verdict = "STATIC<=AUTO"
    else:
        ds = da = 0.0
        verdict = "NO_DATA"
    return {"model": model_name, "rows": rows,
            "mean_drift_static_alpha_pos": ds,
            "mean_drift_auto_alpha_pos": da,
            "verdict": verdict}


def render_table(agg) -> str:
    lines = []
    lines.append("| alpha | n | static_nll | auto_nll | drift_static | drift_auto | Δ(auto-static) |")
    lines.append("|------:|--:|-----------:|---------:|-------------:|-----------:|---------------:|")
    for r in agg["rows"]:
        delta = r["drift_auto_mean"] - r["drift_static_mean"]
        lines.append(
            f"| {r['alpha']:.2f} | {r['n']} | {r['static_nll_mean']:+.4f} | "
            f"{r['auto_nll_mean']:+.4f} | {r['drift_static_mean']:+.4f} | "
            f"{r['drift_auto_mean']:+.4f} | {delta:+.4f} |"
        )
    return "\n".join(lines)


def write_report(out_dir: Path, model_name: str, cmdline: str, git_rev: str,
                 cells, agg, profile_artifact: dict | None,
                 candidates_log: list[dict]):
    md = []
    md.append(f"# S-7 — U-LOPI cross-arch MPS pilot\n")
    md.append(f"**Model loaded**: `{model_name}`\n")
    md.append(f"**Cmdline**: `{cmdline}`\n")
    md.append(f"**Git rev**: `{git_rev}`\n")
    md.append(f"**Generated**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    md.append("")
    md.append("## Candidate models (load attempts)\n")
    for c in candidates_log:
        md.append(f"- `{c['name']}` → **{c['status']}** {c.get('detail','')}")
    md.append("")
    md.append("## Profile artifact (auto mode)\n")
    if profile_artifact:
        md.append(f"- `mu_arch` = **{profile_artifact['mu_arch']}**")
        md.append(f"- `eta_sigma` = **{profile_artifact['eta_sigma']:.4f}**")
        md.append(f"- `num_layers` = {len(profile_artifact['mu_base'])}")
        md.append(f"- `profile_corpus_sha` = `{profile_artifact['profile_corpus_sha']}`")
        md.append(f"- See `profile.json` for full `mu_base` / `sigma_base` arrays.")
    md.append("")
    md.append("## Paired drift table (mean over seeds)\n")
    md.append(render_table(agg))
    md.append("")
    md.append("## Verdict\n")
    md.append(f"- mean drift static (alpha>0) = **{agg['mean_drift_static_alpha_pos']:+.4f} nats**")
    md.append(f"- mean drift auto   (alpha>0) = **{agg['mean_drift_auto_alpha_pos']:+.4f} nats**")
    md.append(f"- **Verdict**: `{agg['verdict']}`")
    md.append("")
    md.append("## Limitations\n")
    md.append("- MPS small-model pilot; flagship cross-arch on GB10 deferred.")
    md.append("- 8 prompts × ≤3 seeds × ≤3 alphas — small-N; treated as a")
    md.append("  smoke / hypothesis check, not a publication-grade result.")
    md.append("- The `LlamaAdapter` claims Qwen2/Qwen2.5/Llama/Mistral. ")
    md.append("  Bit-equal at α=0 is asserted at runtime as the only ")
    md.append("  cross-mode safety check.")
    (out_dir / "REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Primary HF model id; falls back through the candidate list on failure.")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--alpha", default="0,2,5",
                    help="Comma-separated alpha values; alpha=0 is the bit-equal sanity check.")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--out", default="reports/cleanroom/ulopi_xarch/")
    args = ap.parse_args()

    # Red line: no gradient anywhere outside bank write (write_fact uses
    # torch.no_grad internally).
    torch.set_grad_enabled(False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmdline = "python " + " ".join(sys.argv)
    import subprocess
    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        git_rev = "unknown"

    alphas = parse_csv_floats(args.alpha)
    seeds = parse_csv_ints(args.seeds)
    prompts = list(EVAL_PROMPTS)[: args.n_prompts]
    dtype = _dtype(args.dtype)

    # Try the user model first, then the documented fallback chain.
    candidates = [args.model]
    fallbacks = ["Qwen/Qwen2.5-0.5B-Instruct",
                 "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    for f in fallbacks:
        if f not in candidates:
            candidates.append(f)

    candidates_log = []
    loaded = None
    for name in candidates:
        rec = _try_load(name, args.device, dtype)
        if rec is None:
            candidates_log.append({"name": name, "status": "FAILED", "detail": "load error"})
            continue
        try:
            tok, model = rec
            # Verify patcher can attach (would raise if no adapter matches).
            _ = AttnNativePatcher(model)
            loaded = (name, tok, model)
            candidates_log.append({"name": name, "status": "LOADED", "detail": ""})
            break
        except Exception as exc:
            candidates_log.append({"name": name, "status": "PATCH-FAIL",
                                   "detail": f"({exc!r})"})
            print(f"[patch-fail] {name}: {exc!r}", flush=True)
            del rec
            continue

    if loaded is None:
        print("[fatal] no candidate model loaded on this machine.", flush=True)
        write_report(out_dir, model_name="<none>", cmdline=cmdline,
                     git_rev=git_rev, cells=[],
                     agg={"rows": [], "mean_drift_static_alpha_pos": 0.0,
                          "mean_drift_auto_alpha_pos": 0.0, "verdict": "BLOCKED",
                          "model": "<none>"},
                     profile_artifact=None, candidates_log=candidates_log)
        (out_dir / "AGGREGATE.json").write_text(
            json.dumps({"status": "BLOCKED", "candidates": candidates_log},
                       indent=2), encoding="utf-8")
        return 2

    model_name, tok, model = loaded
    print(f"[ok] using {model_name}", flush=True)

    # Baseline (no patch) NLL on the held-out set — the drift reference.
    base_nll_list = _baseline_nll(model, tok, prompts)
    base_nll = float(sum(base_nll_list) / len(base_nll_list))
    print(f"[base] mean NLL = {base_nll:.4f} nats", flush=True)

    cells = []
    profile_artifact = None
    for alpha in alphas:
        for seed in seeds:
            t0 = time.time()
            try:
                cell = run_cell(model_name, model, tok, alpha, seed, prompts)
            except AssertionError:
                raise
            except Exception as exc:
                print(f"[cell-fail] alpha={alpha} seed={seed}: {exc!r}", flush=True)
                traceback.print_exc()
                continue
            cell["base_nll"] = base_nll
            cell["base_nll_list"] = base_nll_list
            cell["model"] = model_name
            cells.append(cell)
            if profile_artifact is None and "mu_base" in cell:
                profile_artifact = {
                    "mu_arch": cell["mu_arch"],
                    "eta_sigma": cell["eta_sigma"],
                    "mu_base": cell["mu_base"],
                    "sigma_base": cell["sigma_base"],
                    "profile_corpus_sha": cell["profile_corpus_sha"],
                }
            dt = time.time() - t0
            print(f"[cell] alpha={alpha:.2f} seed={seed}  "
                  f"static_nll={cell['static_nll']:.4f}  "
                  f"auto_nll={cell['auto_nll']:.4f}  "
                  f"drift_static={cell['static_nll']-base_nll:+.4f}  "
                  f"drift_auto={cell['auto_nll']-base_nll:+.4f}  "
                  f"({dt:.1f}s)", flush=True)
            # Per-cell paired JSON (one file per (alpha, seed)).
            short = model_name.replace("/", "_")
            (out_dir / f"cell_{short}_a{alpha}_s{seed}.json").write_text(
                json.dumps(cell, indent=2), encoding="utf-8")

    agg = aggregate(cells, model_name)
    (out_dir / "AGGREGATE.json").write_text(
        json.dumps({"agg": agg, "cells": cells,
                    "candidates": candidates_log,
                    "git_rev": git_rev, "cmdline": cmdline},
                   indent=2), encoding="utf-8")
    if profile_artifact is not None:
        (out_dir / "profile.json").write_text(
            json.dumps(profile_artifact, indent=2), encoding="utf-8")
    md_table = render_table(agg)
    (out_dir / "AGGREGATE.md").write_text(
        f"# S-7 aggregate — {model_name}\n\n"
        f"Verdict: `{agg['verdict']}`  "
        f"(mean drift static={agg['mean_drift_static_alpha_pos']:+.4f}, "
        f"auto={agg['mean_drift_auto_alpha_pos']:+.4f})\n\n"
        + md_table + "\n", encoding="utf-8")
    write_report(out_dir, model_name, cmdline, git_rev, cells, agg,
                 profile_artifact, candidates_log)
    print(f"[done] verdict = {agg['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
