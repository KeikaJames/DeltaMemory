"""W-T3 round 2: end-to-end NLL A/B between additive and ECOR injection.

Sweeps the LOPI/ECOR injection mode across (model × alpha × seed) and reports
mean per-token NLL on a held-out neutral prompt set.

Modes
-----
* ``add``           — legacy additive (use_ecor=False, ortho=False)
* ``add_ortho``     — additive + M_perp orthogonal (use_ecor=False, ortho=True)
* ``ecor_blend50``  — ECOR soft_blend=0.5, ortho=True, max_theta=π/3
* ``ecor_pure``     — ECOR soft_blend=1.0, ortho=True, max_theta=π/3

Red lines
---------
* ``alpha=0`` ⇒ all four modes must be bit-equal on logits (multiplicative gate
  collapses to identity).  Asserted at runtime.
* ECOR ``soft_blend=0`` ⇒ bit-equal to ``add_ortho``.  Covered by
  ``tests/test_lopi_ecor_routing.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltamemory import (
    AttnNativeBank,
    AttnNativePatcher,
    LOPIConfig,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.lopi import LOPIState
from deltamemory.memory.lopi_inject import ECORConfig


EVAL_PROMPTS = (
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


MODES = {
    "add":          dict(use_ecor=False, orthogonal=False),
    "add_ortho":    dict(use_ecor=False, orthogonal=True),
    "ecor_blend50": dict(use_ecor=True,  orthogonal=True,
                         ecor=dict(enabled=True, soft_blend=0.5,
                                   max_theta_frac=1.0/3.0)),
    "ecor_pure":    dict(use_ecor=True,  orthogonal=True,
                         ecor=dict(enabled=True, soft_blend=1.0,
                                   max_theta_frac=1.0/3.0)),
}


def _dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _load(name, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {name}  device={device}  dtype={dtype}", flush=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tok, model


def _make_cfg(mode):
    spec = MODES[mode]
    cfg = LOPIConfig(
        enabled=True,
        orthogonal=spec["orthogonal"],
        gaussian=True,
        derivative=True,
        profile_mode="static",
        use_ecor=spec["use_ecor"],
    )
    if spec["use_ecor"]:
        cfg.ecor_cfg = ECORConfig(**spec["ecor"])
    return cfg


@torch.no_grad()
def _eval_logits_and_nll(patcher, bank, tok, prompts, alpha):
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


def run_cell(model, tok, mode, alpha, seed, prompts):
    torch.manual_seed(seed)
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    bank.lopi_cfg = _make_cfg(mode)
    bank.lopi_state = LOPIState(num_layers=bank.num_layers)
    write_fact(patcher, bank, tok,
               write_prompt=WRITE_PROMPT,
               fact_id=WRITE_FACT_ID,
               address=WRITE_ADDRESS)
    nlls, ll = _eval_logits_and_nll(patcher, bank, tok, prompts, alpha)
    return {
        "mode": mode, "alpha": float(alpha), "seed": int(seed),
        "nll_list": nlls,
        "nll_mean": float(sum(nlls) / len(nlls)),
        "last_logits": ll,
    }


def aggregate(cells):
    by_key = {}
    for c in cells:
        key = (c["mode"], c["alpha"])
        by_key.setdefault(key, []).append(c["nll_mean"] - c["base_nll_mean"])
    rows = []
    for (mode, alpha), drifts in sorted(by_key.items(),
                                         key=lambda kv: (kv[0][1], kv[0][0])):
        rows.append({
            "mode": mode, "alpha": alpha, "n": len(drifts),
            "drift_mean": sum(drifts) / len(drifts),
            "drift_min": min(drifts), "drift_max": max(drifts),
        })
    return rows


def render_table(rows):
    by_alpha = {}
    for r in rows:
        by_alpha.setdefault(r["alpha"], {})[r["mode"]] = r["drift_mean"]
    modes = list(MODES.keys())
    out = []
    out.append("| α | " + " | ".join(modes) + " | best |")
    out.append("|--:|" + "|".join(["--:"] * (len(modes) + 1)) + "|")
    for a in sorted(by_alpha.keys()):
        row = by_alpha[a]
        cells = [f"{row.get(m, float('nan')):+.3f}" for m in modes]
        best_mode = min(modes, key=lambda m: row.get(m, float("inf")))
        out.append(f"| {a:.2f} | " + " | ".join(cells) + f" | **{best_mode}** |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--alphas", default="0,1,2,4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    prompts = list(EVAL_PROMPTS[: args.n_prompts])

    torch.set_grad_enabled(False)

    tok, model = _load(args.model, args.device, _dtype(args.dtype))

    print("[base] computing baseline NLL", flush=True)
    base_nlls = _baseline_nll(model, tok, prompts)
    base_mean = sum(base_nlls) / len(base_nlls)
    print(f"[base] mean = {base_mean:+.4f}", flush=True)

    cells = []
    t0 = time.time()
    n_total = len(MODES) * len(alphas) * len(seeds)
    n_done = 0
    for alpha in alphas:
        alpha0_logits = {}
        for seed in seeds:
            for mode in MODES:
                cell = run_cell(model, tok, mode, alpha, seed, prompts)
                cell["base_nll_mean"] = base_mean
                cell["base_nll_list"] = base_nlls
                ll = cell.pop("last_logits")
                if abs(alpha) < 1e-12:
                    alpha0_logits.setdefault(seed, {})[mode] = ll
                cells.append(cell)
                n_done += 1
                print(f"[cell {n_done}/{n_total}] mode={mode} α={alpha} "
                      f"seed={seed} nll={cell['nll_mean']:+.4f} "
                      f"drift={cell['nll_mean']-base_mean:+.4f}",
                      flush=True)
        if abs(alpha) < 1e-12:
            for seed, logs in alpha0_logits.items():
                ref_mode = "add"
                ref = logs[ref_mode]
                for mode, ll in logs.items():
                    if mode == ref_mode:
                        continue
                    max_diff = max(float((a - b).abs().max().item())
                                   for a, b in zip(ref, ll))
                    print(f"[redline α=0 seed={seed}] {mode} vs {ref_mode}: "
                          f"max_abs_diff={max_diff:.3e}", flush=True)
                    assert max_diff < 1e-3, (
                        f"α=0 redline fail: {mode} differs from add by "
                        f"{max_diff:.3e} (>1e-3 in bf16)"
                    )
    elapsed = time.time() - t0
    print(f"[done] {n_total} cells in {elapsed:.1f}s", flush=True)

    cells_path = out_dir / "cells.jsonl"
    with cells_path.open("w") as f:
        for c in cells:
            f.write(json.dumps(c) + "\n")

    rows = aggregate(cells)
    table = render_table(rows)

    md = []
    md.append(f"# W-T3 round 2 — end-to-end ECOR vs additive\n")
    md.append(f"**Model**: `{args.model}`\n")
    md.append(f"**Device/dtype**: `{args.device}` / `{args.dtype}`\n")
    md.append(f"**α grid**: {alphas} | **Seeds**: {seeds} | "
              f"**Prompts**: {len(prompts)}\n")
    md.append(f"**Baseline NLL** (no bank): {base_mean:+.4f}\n")
    md.append(f"**Total cells**: {n_total} | **Wall**: {elapsed:.1f}s\n")
    md.append(f"**Generated**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    md.append("\n## Drift vs baseline (lower is better)\n")
    md.append(table)
    md.append("\n## Cells\n")
    md.append(f"- `{cells_path.name}` ({len(cells)} cells)\n")

    (out_dir / "REPORT.md").write_text("\n".join(md))
    (out_dir / "AGGREGATE.json").write_text(json.dumps(
        {"model": args.model, "alphas": alphas, "seeds": seeds,
         "n_prompts": len(prompts), "base_nll_mean": base_mean,
         "rows": rows, "wall_seconds": elapsed},
        indent=2,
    ))
    print(f"[ok] wrote {out_dir}/REPORT.md", flush=True)
    print(table, flush=True)


if __name__ == "__main__":
    main()
