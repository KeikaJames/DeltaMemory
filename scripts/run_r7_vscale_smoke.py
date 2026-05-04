#!/usr/bin/env python3
"""R-7: bank-side V-scale calibration smoke.

Compares `value_scale_mode=none` against the v3.6 default
`auto_rms_cap` on a small no-v_norm family model.  The goal is not a
publication-grade sweep; it is a fast sanity check that the new schema knob
actually changes captured bank M_V magnitude and preserves alpha=0 safety.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deltamemory.memory.attn_native_bank import AttnNativePatcher, fresh_bank, write_fact
from deltamemory.memory.lopi import LOPIConfig, LOPIState


EVAL_PROMPTS = (
    "The boiling point of water at sea level is one hundred degrees Celsius.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The Pacific Ocean covers about one third of Earth's surface.",
    "DNA contains adenine, thymine, guanine, and cytosine.",
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


def _csv_floats(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def _csv_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _csv_strs(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    with torch.no_grad():
        ids = tok("hello", return_tensors="pt").input_ids.to(device)
        model(input_ids=ids, use_cache=False)
    return tok, model


@torch.no_grad()
def _seq_nll(model, tok, prompt: str) -> tuple[float, torch.Tensor]:
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    am = enc["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=am, use_cache=False)
    logits = out.logits[0]
    targets = ids[0, 1:]
    logp = F.log_softmax(logits[:-1].float(), dim=-1)
    nll = -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    last = int(am.sum(dim=1).item() - 1)
    return float(nll.mean().item()), logits[last].float().detach().cpu()


@torch.no_grad()
def _seq_nll_with_bank(patcher, bank, tok, prompt: str, alpha: float) -> tuple[float, torch.Tensor]:
    with patcher.patched(), patcher.injecting(bank, alpha=alpha):
        return _seq_nll(patcher.model, tok, prompt)


def _mean_bank_v_rms(bank) -> float:
    vals: list[float] = []
    for v in bank.M_V:
        if v.numel() == 0:
            continue
        rms = torch.linalg.vector_norm(v.float(), ord=2, dim=-1) / (v.size(-1) ** 0.5)
        vals.append(float(rms.mean().item()))
    return sum(vals) / max(len(vals), 1)


def _configure_lopi(bank) -> None:
    bank.lopi_cfg = LOPIConfig(
        enabled=True,
        orthogonal=False,
        gaussian=True,
        derivative=True,
        profile_mode="static",
    )
    bank.lopi_state = LOPIState(num_layers=bank.num_layers)


def run_cell(model, tok, value_scale_mode: str, alpha: float, seed: int, prompts: list[str]) -> dict:
    torch.manual_seed(seed)
    patcher = AttnNativePatcher(model)
    bank = fresh_bank(model)
    bank.value_scale_mode = value_scale_mode
    _configure_lopi(bank)
    write_fact(
        patcher,
        bank,
        tok,
        write_prompt=WRITE_PROMPT,
        fact_id=WRITE_FACT_ID,
        address=WRITE_ADDRESS,
    )

    base_nlls, inj_nlls = [], []
    max_alpha0_diff = 0.0
    for prompt in prompts:
        base_nll, base_last = _seq_nll(model, tok, prompt)
        inj_nll, inj_last = _seq_nll_with_bank(patcher, bank, tok, prompt, alpha=alpha)
        base_nlls.append(base_nll)
        inj_nlls.append(inj_nll)
        if abs(alpha) < 1e-12:
            max_alpha0_diff = max(max_alpha0_diff, float((base_last - inj_last).abs().max().item()))

    if abs(alpha) < 1e-12:
        assert max_alpha0_diff < 1e-6, f"alpha=0 bit-equal failed: {max_alpha0_diff:.3e}"

    base = sum(base_nlls) / len(base_nlls)
    inj = sum(inj_nlls) / len(inj_nlls)
    return {
        "value_scale_mode": value_scale_mode,
        "alpha": float(alpha),
        "seed": int(seed),
        "base_nll": float(base),
        "inj_nll": float(inj),
        "nll_drift": float(inj - base),
        "mean_bank_v_rms": float(_mean_bank_v_rms(bank)),
        "alpha0_max_abs_diff": float(max_alpha0_diff) if abs(alpha) < 1e-12 else None,
    }


def aggregate(cells: list[dict]) -> dict:
    grouped: dict[tuple[str, float], list[dict]] = {}
    for c in cells:
        grouped.setdefault((c["value_scale_mode"], c["alpha"]), []).append(c)
    rows = []
    for (mode, alpha), group in sorted(grouped.items()):
        rows.append({
            "value_scale_mode": mode,
            "alpha": alpha,
            "n": len(group),
            "mean_drift": sum(c["nll_drift"] for c in group) / len(group),
            "mean_bank_v_rms": sum(c["mean_bank_v_rms"] for c in group) / len(group),
        })
    return {"rows": rows}


def write_report(out_dir: Path, model_name: str, cells: list[dict], agg: dict, cmdline: str) -> None:
    lines = [
        "# R-7 V-scale calibration smoke",
        "",
        f"**Model**: `{model_name}`",
        f"**Generated**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"**Cmdline**: `{cmdline}`",
        "",
        "| value_scale_mode | alpha | n | mean drift | mean bank V RMS |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in agg["rows"]:
        lines.append(
            f"| `{r['value_scale_mode']}` | {r['alpha']:.2f} | {r['n']} | "
            f"{r['mean_drift']:+.4f} | {r['mean_bank_v_rms']:.4f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "`auto_rms_cap` should keep alpha=0 bit-equal and cap no-v_norm family "
        "bank values at the configured per-head RMS without amplifying already "
        "small V activations. Drift is a smoke signal only; full R-7 resweep "
        "still needs the R-4/R-5.2 grids.",
        "",
        f"Raw cells: `{(out_dir / 'cells.json').as_posix()}`.",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--alphas", default="0,2")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-prompts", type=int, default=2)
    ap.add_argument("--value-scale-modes", default="none,auto_rms_cap")
    ap.add_argument("--out", default="reports/cleanroom/r7_vscale_smoke")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = list(EVAL_PROMPTS[: max(1, min(args.n_prompts, len(EVAL_PROMPTS)))])
    alphas = _csv_floats(args.alphas)
    seeds = _csv_ints(args.seeds)
    modes = _csv_strs(args.value_scale_modes)

    tok, model = _load(args.model, args.device, _dtype(args.dtype))
    cells = []
    for mode in modes:
        for alpha in alphas:
            for seed in seeds:
                c = run_cell(model, tok, mode, alpha, seed, prompts)
                cells.append(c)
                print(
                    f"[cell] mode={mode} alpha={alpha} seed={seed} "
                    f"drift={c['nll_drift']:+.4f} v_rms={c['mean_bank_v_rms']:.4f}",
                    flush=True,
                )
                (out_dir / "cells.json").write_text(json.dumps(cells, indent=2), encoding="utf-8")

    agg = aggregate(cells)
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    write_report(out_dir, args.model, cells, agg, " ".join(sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
