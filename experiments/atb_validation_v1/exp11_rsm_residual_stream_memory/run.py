"""Exp11 — Residual Stream Memory (RSM).

RSM captures one residual-stream vector per layer from each write prompt and
replays selected memories into the read prompt after a two-pass max-layer cosine
gate.  This experiment tests whether residual replay separates correct memories
from random / shuffled / gate-off controls better than ANB-style KV injection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deltamemory.memory.rsm_injector import RSMConfig, RSMInjector, RSMMemoryBank
from experiments.atb_validation_v1._lib import (
    evaluate_prompt,
    filter_cf_for_tokenizer,
    first_token_id,
    load_counterfact,
    load_model,
    neutral_prompts,
    seed_everything,
)
from experiments.atb_validation_v1._lib.aggregator import aggregate
from experiments.atb_validation_v1._lib.cf_runner import build_write_prompt, render_query
from experiments.atb_validation_v1._lib.manifest import write_manifest

EXPERIMENT = "exp11_rsm_residual_stream_memory"
RSM_VARIANTS = ["correct_memory", "random_memory", "shuffled_layers", "gate_off"]
_KL_LAST = 8


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def _sha1(path: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def _eligible_rows(counterfact_path: Path, tok: Any, n_prompts: int | None) -> list[dict]:
    raw = load_counterfact(counterfact_path)
    kept, dropped = filter_cf_for_tokenizer(raw, tok)
    _log(f"counterfact kept={len(kept)} dropped={dropped}")
    if n_prompts is not None:
        kept = kept[:n_prompts]
        _log(f"truncated to n_prompts={len(kept)}")
    eligible = []
    for row in kept:
        wp = build_write_prompt(row, row["target_new"])
        if wp is None:
            continue
        row = dict(row)
        row["_write_prompt"] = wp
        eligible.append(row)
    _log(f"eligible with write prompt={len(eligible)}")
    return eligible


def _tokenize(tok: Any, text: str, device: str, max_length: int = 256) -> tuple[torch.Tensor, torch.Tensor | None]:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    ids = enc["input_ids"].to(device)
    mask = enc.get("attention_mask")
    if mask is not None:
        mask = mask.to(device)
    return ids, mask


def _build_memory_cache(
    rsm: RSMInjector,
    tok: Any,
    rows: list[dict],
    device: str,
) -> dict[str, torch.Tensor]:
    cache: dict[str, torch.Tensor] = {}
    for idx, row in enumerate(rows, start=1):
        fid = str(row["id"])
        ids, mask = _tokenize(tok, row["_write_prompt"], device)
        cache[fid] = rsm.capture(ids, mask)
        if idx % 50 == 0:
            _log(f"captured {idx}/{len(rows)} RSM memories")
    return cache


def _distractor_ids(rows: list[dict], current_id: str, bank_size: int, seed: int) -> list[str]:
    ids = [str(r["id"]) for r in rows if str(r["id"]) != current_id]
    stable_id = sum((idx + 1) * ord(ch) for idx, ch in enumerate(current_id))
    rng = random.Random((seed + 1) * 1_000_003 + stable_id)
    rng.shuffle(ids)
    return ids[:max(0, bank_size)]


def _make_bank(
    *,
    variant: str,
    row: dict,
    rows: list[dict],
    cache: dict[str, torch.Tensor],
    bank_size: int,
    seed: int,
) -> RSMMemoryBank:
    current_id = str(row["id"])
    if variant == "random_memory":
        fact_ids = _distractor_ids(rows, current_id, bank_size, seed)
    else:
        fact_ids = [current_id] + _distractor_ids(rows, current_id, bank_size - 1, seed)
    if not fact_ids:
        raise ValueError(f"RSM variant={variant} requires at least one memory.")
    memories = torch.stack([cache[fid] for fid in fact_ids], dim=0)
    bank = RSMMemoryBank(memories=memories, fact_ids=fact_ids)
    if variant == "shuffled_layers":
        bank = bank.shuffled_layers(seed=seed + 0x51A7)
    return bank


def _make_neutral_bank(
    *,
    variant: str,
    rows: list[dict],
    cache: dict[str, torch.Tensor],
    bank_size: int,
    seed: int,
) -> RSMMemoryBank:
    if not rows:
        raise ValueError("neutral drift requires at least one cached memory.")
    rng = random.Random(seed + 0xD41F7)
    ids = [str(r["id"]) for r in rows]
    rng.shuffle(ids)
    fact_ids = ids[:max(1, min(bank_size, len(ids)))]
    bank = RSMMemoryBank(torch.stack([cache[fid] for fid in fact_ids], dim=0), fact_ids)
    if variant == "shuffled_layers":
        bank = bank.shuffled_layers(seed=seed + 0x51A7)
    return bank


@torch.no_grad()
def _last_k_logsoftmax_model(model: Any, tok: Any, prompt: str, device: str) -> torch.Tensor:
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=True)], device=device)
    out = model(input_ids=ids, use_cache=False)
    return F.log_softmax(out.logits[0, -_KL_LAST:].float(), dim=-1).detach().cpu()


@torch.no_grad()
def _last_k_logsoftmax_rsm(
    rsm: RSMInjector,
    bank: RSMMemoryBank,
    tok: Any,
    prompt: str,
    device: str,
) -> torch.Tensor:
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=True)], device=device)
    out, _diag = rsm.forward_with_memory(bank, input_ids=ids)
    return F.log_softmax(out.logits[0, -_KL_LAST:].float(), dim=-1).detach().cpu()


def _js_nats(logp_a: torch.Tensor, logp_b: torch.Tensor) -> float:
    p = torch.exp(logp_a)
    q = torch.exp(logp_b)
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp(min=1e-30))
    return float(0.5 * ((p * (logp_a - log_m)).sum(-1) + (q * (logp_b - log_m)).sum(-1)).mean().item())


def _kl_nats(logp_a: torch.Tensor, logp_b: torch.Tensor) -> float:
    p = torch.exp(logp_a)
    return float((p * (logp_a - logp_b)).sum(-1).mean().item())


def _compute_rsm_neutral_drifts(
    *,
    model: Any,
    tok: Any,
    rsm: RSMInjector,
    rows: list[dict],
    cache: dict[str, torch.Tensor],
    device: str,
    eta: float,
    theta: float,
    bank_size: int,
    n_neutral: int,
) -> dict[str, tuple[float | None, float | None]]:
    drifts: dict[str, tuple[float | None, float | None]] = {
        "base_model": (0.0, 0.0),
    }
    if n_neutral <= 0:
        for variant in RSM_VARIANTS:
            drifts[variant] = (None, None)
        return drifts
    prompts = neutral_prompts(n=n_neutral)
    base_lps = [_last_k_logsoftmax_model(model, tok, prompt, device) for prompt in prompts]
    for variant in RSM_VARIANTS:
        rsm.config = RSMConfig(eta=eta, theta=theta, gate_off=(variant == "gate_off"))
        bank = _make_neutral_bank(
            variant=variant,
            rows=rows,
            cache=cache,
            bank_size=bank_size,
            seed=0,
        )
        js_vals: list[float] = []
        kl_vals: list[float] = []
        for prompt, base_lp in zip(prompts, base_lps):
            inj_lp = _last_k_logsoftmax_rsm(rsm, bank, tok, prompt, device)
            js_vals.append(_js_nats(base_lp, inj_lp))
            kl_vals.append(_kl_nats(base_lp, inj_lp))
        drifts[variant] = (sum(js_vals) / len(js_vals), sum(kl_vals) / len(kl_vals))
    return drifts


def _phase_b_verdict(best: dict[str, float]) -> str:
    correct = best.get("correct_memory", float("-inf"))
    base = best.get("base_model", float("inf"))
    random_memory = best.get("random_memory", float("inf"))
    gap = best.get("gap", float("-inf"))
    if gap > 0 and correct > base and correct > random_memory:
        return "PASS_DIRECTIONAL"
    if correct > base and correct > random_memory:
        return "STABILIZER_ONLY"
    return "FAIL"


@torch.no_grad()
def _continuation_logp_rsm(
    rsm: RSMInjector,
    bank: RSMMemoryBank,
    tok: Any,
    prompt: str,
    target: str,
    device: str,
) -> tuple[float, list[int], dict[str, Any]]:
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    sep = "" if (prompt.endswith(" ") or not prompt) else " "
    full_ids = tok.encode(prompt + sep + target, add_special_tokens=True)
    if len(full_ids) <= len(prompt_ids):
        return float("nan"), [], {}
    target_ids = full_ids[len(prompt_ids):]
    prompt_tensor = torch.tensor([prompt_ids], device=device)
    prompt_scores = rsm.score(bank, input_ids=prompt_tensor)
    total = 0.0
    first_diag: dict[str, Any] = {}
    for i, tid in enumerate(target_ids):
        prefix_ids = full_ids[: len(prompt_ids) + i]
        ids = torch.tensor([prefix_ids], device=device)
        out, diag = rsm.forward_with_scores(bank, prompt_scores, input_ids=ids)
        if i == 0:
            first_diag = diag
        logits = out.logits[0, -1].float()
        logp = F.log_softmax(logits, dim=-1)
        total += float(logp[tid].item())
    return total, target_ids, first_diag


@torch.no_grad()
def _first_token_rank_rsm(
    rsm: RSMInjector,
    bank: RSMMemoryBank,
    tok: Any,
    prompt: str,
    target_first_id: int,
    device: str,
) -> int:
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=True)], device=device)
    scores = rsm.score(bank, input_ids=ids)
    out, _diag = rsm.forward_with_scores(bank, scores, input_ids=ids)
    logits = out.logits[0, -1].float()
    sorted_ids = torch.argsort(logits, descending=True)
    return int((sorted_ids == target_first_id).nonzero(as_tuple=False).item())


def _evaluate_rsm(
    rsm: RSMInjector,
    bank: RSMMemoryBank,
    tok: Any,
    row: dict,
    device: str,
) -> dict[str, Any]:
    query = render_query(row)
    logp_new, ids_new, diag = _continuation_logp_rsm(
        rsm, bank, tok, query, row["target_new"], device
    )
    logp_true, _ids_true, _diag_true = _continuation_logp_rsm(
        rsm, bank, tok, query, row["target_true"], device
    )
    target_new_first = ids_new[0] if ids_new else first_token_id(tok, query, row["target_new"])
    rank = _first_token_rank_rsm(rsm, bank, tok, query, target_new_first, device)
    current_id = str(row["id"])
    return {
        "target_new_logprob": logp_new,
        "target_true_logprob": logp_true,
        "margin": logp_new - logp_true,
        "target_rank": rank,
        "recall_at_1": rank == 0,
        "rsm_activation_rate": diag.get("rsm_activation_rate"),
        "rsm_max_score": diag.get("rsm_max_score"),
        "rsm_top_fact_id": diag.get("rsm_top_fact_id"),
        "rsm_top_memory_hit": diag.get("rsm_top_fact_id") == current_id,
    }


def _run_one_config(
    *,
    model: Any,
    tok: Any,
    rsm: RSMInjector,
    rows: list[dict],
    cache: dict[str, torch.Tensor],
    device: str,
    out_dir: Path,
    eta: float,
    theta: float,
    bank_size: int,
    seeds: list[int],
    include_anb_best: bool,
    n_neutral: int = 0,
) -> Path:
    if include_anb_best:
        raise RuntimeError(
            "--include-anb-best is disabled in Exp11 until the full Exp10 A3 "
            "LOPI/mHC VariantContext plumbing is available on this branch."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    drift_by_variant = _compute_rsm_neutral_drifts(
        model=model,
        tok=tok,
        rsm=rsm,
        rows=rows,
        cache=cache,
        device=device,
        eta=eta,
        theta=theta,
        bank_size=bank_size,
        n_neutral=n_neutral,
    )
    with open(results_path, "a") as f:
        for seed in seeds:
            seed_everything(seed)
            for variant in ["base_model", *RSM_VARIANTS]:
                cfg = RSMConfig(eta=eta, theta=theta, gate_off=(variant == "gate_off"))
                rsm.config = cfg
                for row in rows:
                    query = render_query(row)
                    if variant == "base_model":
                        mp = evaluate_prompt(
                            model, tok, query, row["target_new"], row["target_true"], device
                        )
                        extra = {
                            "rsm_activation_rate": math.nan,
                            "rsm_max_score": math.nan,
                            "rsm_top_fact_id": None,
                            "rsm_top_memory_hit": False,
                        }
                    else:
                        bank = _make_bank(
                            variant=variant,
                            row=row,
                            rows=rows,
                            cache=cache,
                            bank_size=bank_size,
                            seed=seed,
                        )
                        mp = _evaluate_rsm(rsm, bank, tok, row, device)
                        extra = {}
                    js_drift, kl_drift = drift_by_variant.get(variant, (None, None))
                    rec = {
                        "experiment": EXPERIMENT,
                        "variant": variant,
                        "method": "none" if variant == "base_model" else "rsm",
                        "alpha": eta,
                        "rsm_eta": eta,
                        "rsm_theta": theta,
                        "rsm_hook_point": "block_output",
                        "seed": seed,
                        "prompt_id": row["id"],
                        "subject": row["subject"],
                        "target_new": row["target_new"],
                        "target_true": row["target_true"],
                        "bank_size": 0 if variant == "base_model" else bank_size,
                        "js_drift": js_drift,
                        "kl_drift": kl_drift,
                        **mp,
                        **extra,
                    }
                    f.write(json.dumps(rec) + "\n")
    return results_path


def _read_margins(summary_path: Path) -> dict[str, float]:
    margins: dict[str, float] = {}
    with open(summary_path) as f:
        for row in csv.DictReader(f):
            try:
                margins[row["variant"]] = float(row["mean_margin"])
            except (KeyError, ValueError):
                pass
    return margins


def _gap(summary_path: Path) -> float:
    m = _read_margins(summary_path)
    correct = m.get("correct_memory", float("-inf"))
    controls = [m.get("random_memory"), m.get("shuffled_layers"), m.get("gate_off")]
    controls = [c for c in controls if c is not None]
    return correct - max(controls) if controls else float("nan")


def _write_config_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    cf_path: Path,
    phase: str,
    eta: float,
    theta: float,
    n_prompts: int,
) -> None:
    write_manifest(
        out_dir,
        experiment=EXPERIMENT,
        repo_root=ROOT,
        dataset_path=cf_path,
        dataset_sha1=_sha1(cf_path),
        model=args.model,
        dtype=args.dtype,
        attention_impl="eager",
        seeds=[int(s) for s in args.seeds.split(",") if s],
        variants=[
            {"name": v, "method": "rsm", "eta": eta, "theta": theta}
            for v in RSM_VARIANTS
        ],
        write_template="Fact: {subject} {phrase} {target_new}.",
        read_template="prompt.format(subject)",
        enabled_modules=["ResidualStreamMemory"],
        disabled_modules=["AttnNativeBank", "SCAR", "CAA"],
        extra={
            "phase": phase,
            "eta": eta,
            "theta": theta,
            "bank_size": args.bank_size,
            "n_prompts": n_prompts,
            "gate": "max_layer_cosine; gate_off skips theta but keeps nonnegative score weights",
            "inject_only_last_token": True,
            "include_anb_best": bool(args.include_anb_best),
            "n_neutral": args.n_neutral,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Exp11: Residual Stream Memory")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--counterfact", default=str(ROOT / "experiments/datasets/counterfact_1k.jsonl"))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--eta-grid", default="0.02,0.05,0.10,0.20")
    p.add_argument("--theta-grid", default="0.30,0.50,0.70")
    p.add_argument("--bank-size", type=int, default=200)
    p.add_argument("--n-prompts-smoke", type=int, default=100)
    p.add_argument("--n-prompts-confirm", type=int, default=807)
    p.add_argument("--n-neutral", type=int, default=100)
    p.add_argument("--phase", default="AB", choices=["A", "B", "AB"])
    p.add_argument(
        "--include-anb-best",
        action="store_true",
        help="Disabled for now: Exp10 A3 requires full LOPI/mHC VariantContext plumbing.",
    )
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    cf_path = Path(args.counterfact)

    _log(f"loading {args.model} dtype={args.dtype} device={args.device}")
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)
    rsm = RSMInjector(model, RSMConfig())

    phase_a_scores: list[dict[str, Any]] = []
    if "A" in args.phase:
        rows = _eligible_rows(cf_path, tok, args.n_prompts_smoke)
        cache = _build_memory_cache(rsm, tok, rows, args.device)
        for eta_s in args.eta_grid.split(","):
            for theta_s in args.theta_grid.split(","):
                eta, theta = float(eta_s), float(theta_s)
                tag = f"eta_{eta:.2f}_theta_{theta:.2f}".replace(".", "_")
                cfg_out = out_root / "phase_a" / tag
                _log(f"phase A config {tag}")
                _write_config_manifest(
                    cfg_out, args=args, cf_path=cf_path, phase="A",
                    eta=eta, theta=theta, n_prompts=len(rows),
                )
                res = _run_one_config(
                    model=model,
                    tok=tok,
                    rsm=rsm,
                    rows=rows,
                    cache=cache,
                    device=args.device,
                    out_dir=cfg_out,
                    eta=eta,
                    theta=theta,
                    bank_size=args.bank_size,
                    seeds=seeds,
                    include_anb_best=False,
                    n_neutral=0,
                )
                summary = aggregate(
                    res,
                    experiment=f"{EXPERIMENT}_a_{tag}",
                    model=args.model,
                    dataset=cf_path.name,
                    out_dir=cfg_out,
                )
                gap = _gap(summary)
                phase_a_scores.append({"tag": tag, "eta": eta, "theta": theta, "gap": gap})
                _log(f"phase A {tag}: gap={gap:.4f}")
        phase_a_scores.sort(key=lambda r: r["gap"], reverse=True)
        (out_root / "phase_a_selection.json").write_text(json.dumps(phase_a_scores[:2], indent=2))

    if "B" in args.phase:
        if phase_a_scores:
            selected = phase_a_scores[:2]
        else:
            selection_path = out_root / "phase_a_selection.json"
            selected = json.loads(selection_path.read_text())[:2]
        rows = _eligible_rows(cf_path, tok, args.n_prompts_confirm)
        cache = _build_memory_cache(rsm, tok, rows, args.device)
        phase_b_scores = []
        for sel in selected:
            eta, theta = float(sel["eta"]), float(sel["theta"])
            tag = f"eta_{eta:.2f}_theta_{theta:.2f}".replace(".", "_")
            cfg_out = out_root / "phase_b" / tag
            _log(f"phase B config {tag}")
            _write_config_manifest(
                cfg_out, args=args, cf_path=cf_path, phase="B",
                eta=eta, theta=theta, n_prompts=len(rows),
            )
            res = _run_one_config(
                model=model,
                tok=tok,
                rsm=rsm,
                rows=rows,
                cache=cache,
                device=args.device,
                out_dir=cfg_out,
                eta=eta,
                theta=theta,
                bank_size=args.bank_size,
                seeds=seeds,
                include_anb_best=args.include_anb_best,
                n_neutral=args.n_neutral,
            )
            summary = aggregate(
                res,
                experiment=f"{EXPERIMENT}_b_{tag}",
                model=args.model,
                dataset=cf_path.name,
                out_dir=cfg_out,
            )
            gap = _gap(summary)
            margins = _read_margins(summary)
            phase_b_scores.append({"tag": tag, "eta": eta, "theta": theta, "gap": gap, **margins})
            _log(f"phase B {tag}: gap={gap:.4f}")
        phase_b_scores.sort(key=lambda r: r["gap"], reverse=True)
        (out_root / "phase_b_summary.json").write_text(json.dumps(phase_b_scores, indent=2))
        best = phase_b_scores[0] if phase_b_scores else {}
        verdict = _phase_b_verdict(best)
        (out_root / "verdict.txt").write_text(verdict + "\n")
        _log(f"verdict={verdict}")


if __name__ == "__main__":
    main()
