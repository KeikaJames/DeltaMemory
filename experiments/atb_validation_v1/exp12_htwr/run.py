"""Exp12 — HTWR (Hard-Top1 Whitened Retrieval).

This runner currently implements **Phase α / Tier 0**: the oracle ceiling and
its controls.  T0 isolates the injection pipeline from retrieval by using a
fact_id lookup (oracle), and provides three controls:

- ``oracle_correct``    : oracle retrieve correct memory, eta from grid, sign +1.
- ``oracle_random``     : oracle retrieves a wrong memory (RandomRetriever).
- ``oracle_shuffled``   : oracle retrieves correct memory, but bank layers
                           are shuffled (norm-preserved, identity-destroyed).
- ``oracle_sign_flip``  : oracle retrieve correct memory, sign = -1.

If oracle_correct beats all three controls and base, the injection path is
healthy and we proceed to Tier 1 (raw cosine retrieval).  If even oracle
fails, residual-stream injection has no useful signal and we halt.

Single-config, single-hook-point smoke design.  Sweeps go via re-invocation.
"""
from __future__ import annotations

import argparse
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

from deltamemory.memory.htwr_injector import (
    HTWRConfig,
    HTWRInjector,
    HTWRMemoryBank,
    OracleRetriever,
    RandomRetriever,
    WritePrompt,
)
from experiments.atb_validation_v1._lib import (
    evaluate_prompt,
    filter_cf_for_tokenizer,
    first_token_id,
    load_counterfact,
    load_model,
    seed_everything,
)
from experiments.atb_validation_v1._lib.aggregator import aggregate
from experiments.atb_validation_v1._lib.cf_runner import build_write_prompt, render_query
from experiments.atb_validation_v1._lib.manifest import write_manifest

EXPERIMENT = "exp12_htwr"
T0_VARIANTS = ["oracle_correct", "oracle_random", "oracle_shuffled", "oracle_sign_flip"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


def _sha1(path: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def _eligible_rows(cf_path: Path, tok: Any, n_prompts: int | None) -> list[dict]:
    raw = load_counterfact(cf_path)
    kept, dropped = filter_cf_for_tokenizer(raw, tok)
    _log(f"counterfact kept={len(kept)} dropped={dropped}")
    if n_prompts is not None:
        kept = kept[:n_prompts]
    elig: list[dict] = []
    for row in kept:
        wp = build_write_prompt(row, row["target_new"])
        if wp is None:
            continue
        row = dict(row)
        row["_write_prompt"] = wp
        elig.append(row)
    _log(f"eligible={len(elig)}")
    return elig


def _tokenize(tok: Any, text: str, device: str, max_length: int = 256) -> torch.Tensor:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    return enc["input_ids"].to(device)


def _build_bank(
    htwr: HTWRInjector,
    tok: Any,
    rows: list[dict],
    device: str,
) -> HTWRMemoryBank:
    write_prompts = []
    for row in rows:
        ids = _tokenize(tok, row["_write_prompt"], device)
        write_prompts.append(
            WritePrompt(fact_id=str(row["id"]), input_ids=ids, subject=row["subject"])
        )
    # Capture inline (build_bank does the same but we want progress logging).
    mems = []
    for i, wp in enumerate(write_prompts, 1):
        mems.append(htwr.capture(wp.input_ids))
        if i % 25 == 0:
            _log(f"captured {i}/{len(write_prompts)} memories")
    bank = HTWRMemoryBank(
        memories=torch.stack(mems, dim=0),
        fact_ids=[wp.fact_id for wp in write_prompts],
        subjects=[wp.subject for wp in write_prompts],
    )
    return bank


def _subbank_for_row(
    *,
    variant: str,
    bank: HTWRMemoryBank,
    row: dict,
    bank_size: int,
    seed: int,
) -> tuple[HTWRMemoryBank, str]:
    """Per-row subbank: correct fact + (bank_size-1) distractors.

    Returns (subbank, correct_fact_id).
    """
    current_id = str(row["id"])
    correct_idx = bank.fact_ids.index(current_id)
    other = [i for i in range(bank.n_memories) if i != correct_idx]
    stable_id = sum((k + 1) * ord(c) for k, c in enumerate(current_id))
    rng = random.Random((seed + 1) * 1_000_003 + stable_id)
    rng.shuffle(other)
    pick = [correct_idx] + other[: max(0, bank_size - 1)]
    sub_mem = bank.memories[pick]
    sub_ids = [bank.fact_ids[i] for i in pick]
    sub_subj = [bank.subjects[i] for i in pick] if bank.subjects else None
    sub = HTWRMemoryBank(memories=sub_mem, fact_ids=sub_ids, subjects=sub_subj)
    if variant == "oracle_shuffled":
        sub = sub.shuffled_layers(seed=seed + 0x51A7)
    return sub, current_id


@torch.no_grad()
def _evaluate_htwr(
    htwr: HTWRInjector,
    bank: HTWRMemoryBank,
    correct_fact_id: str,
    retriever: Any,
    tok: Any,
    row: dict,
    device: str,
) -> dict[str, Any]:
    """Single-row evaluation: retrieve top-1, inject, score continuations."""
    query = render_query(row)
    q_ids = _tokenize(tok, query, device)
    # Retrieval is computed once on the query prompt itself.
    q_res = htwr.query_residuals(q_ids)
    retrieval = retriever.retrieve(q_res, bank, correct_fact_id=correct_fact_id)
    chosen_mem = bank.index(retrieval.top_index)

    # Continuation log-prob using teacher forcing for both targets.
    def _cont_logp(target: str) -> tuple[float, list[int]]:
        prompt_ids = tok.encode(query, add_special_tokens=True)
        sep = "" if (query.endswith(" ") or not query) else " "
        full_ids = tok.encode(query + sep + target, add_special_tokens=True)
        if len(full_ids) <= len(prompt_ids):
            return float("nan"), []
        tgt_ids = full_ids[len(prompt_ids):]
        total = 0.0
        for i, tid in enumerate(tgt_ids):
            prefix = full_ids[: len(prompt_ids) + i]
            ids = torch.tensor([prefix], device=device)
            out, _ = htwr.forward_with_memory(chosen_mem, input_ids=ids)
            logits = out.logits[0, -1].float()
            total += float(F.log_softmax(logits, dim=-1)[tid].item())
        return total, tgt_ids

    logp_new, ids_new = _cont_logp(row["target_new"])
    logp_true, _ = _cont_logp(row["target_true"])
    # First-token rank for target_new under injected logits.
    target_new_first = ids_new[0] if ids_new else first_token_id(tok, query, row["target_new"])
    ids = torch.tensor([tok.encode(query, add_special_tokens=True)], device=device)
    out, _ = htwr.forward_with_memory(chosen_mem, input_ids=ids)
    logits = out.logits[0, -1].float()
    sorted_ids = torch.argsort(logits, descending=True)
    rank = int((sorted_ids == target_new_first).nonzero(as_tuple=False).item())

    return {
        "target_new_logprob": logp_new,
        "target_true_logprob": logp_true,
        "margin": logp_new - logp_true,
        "target_rank": rank,
        "recall_at_1": rank == 0,
        "htwr_top_index": retrieval.top_index,
        "htwr_top_fact_id": retrieval.top_fact_id,
        "htwr_top_score": retrieval.top_score,
        "htwr_correct_score": retrieval.correct_score,
        "htwr_retrieval_correct": retrieval.retrieval_accuracy,
        "htwr_top_minus_mean": retrieval.top_minus_mean,
        "htwr_score_std": retrieval.score_std,
    }


def _run_tier0(
    *,
    model: Any,
    tok: Any,
    rows: list[dict],
    full_bank: HTWRMemoryBank,
    device: str,
    out_dir: Path,
    eta: float,
    hook_point: str,
    bank_size: int,
    seeds: list[int],
    inject_layers: tuple[int, ...] | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()
    oracle = OracleRetriever()
    rand_retr = RandomRetriever(seed=0xC0DE)
    with open(results_path, "a") as f:
        for seed in seeds:
            seed_everything(seed)
            for variant in ["base_model", *T0_VARIANTS]:
                # Build injector with appropriate sign.
                inj_sign = -1.0 if variant == "oracle_sign_flip" else 1.0
                cfg = HTWRConfig(
                    eta=eta,
                    hook_point=hook_point,
                    inject_layers=inject_layers,
                    inject_sign=inj_sign,
                )
                htwr = HTWRInjector(model, cfg)
                for row in rows:
                    if variant == "base_model":
                        mp = evaluate_prompt(
                            model, tok, render_query(row),
                            row["target_new"], row["target_true"], device,
                        )
                        extra = {
                            "htwr_top_index": None, "htwr_top_fact_id": None,
                            "htwr_top_score": math.nan, "htwr_correct_score": math.nan,
                            "htwr_retrieval_correct": None,
                            "htwr_top_minus_mean": math.nan, "htwr_score_std": math.nan,
                        }
                    else:
                        sub, cid = _subbank_for_row(
                            variant=variant, bank=full_bank, row=row,
                            bank_size=bank_size, seed=seed,
                        )
                        retr = rand_retr if variant == "oracle_random" else oracle
                        mp = _evaluate_htwr(htwr, sub, cid, retr, tok, row, device)
                        extra = {}
                    rec = {
                        "experiment": EXPERIMENT,
                        "tier": "T0",
                        "variant": variant,
                        "method": "none" if variant == "base_model" else "htwr",
                        "alpha": eta,
                        "htwr_eta": eta,
                        "htwr_hook_point": hook_point,
                        "htwr_inject_sign": inj_sign,
                        "seed": seed,
                        "prompt_id": row["id"],
                        "subject": row["subject"],
                        "target_new": row["target_new"],
                        "target_true": row["target_true"],
                        "bank_size": 0 if variant == "base_model" else bank_size,
                        **mp,
                        **extra,
                    }
                    f.write(json.dumps(rec) + "\n")
    return results_path


def _write_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    cf_path: Path,
    eta: float,
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
            {"name": v, "method": "htwr", "eta": eta, "tier": "T0"}
            for v in T0_VARIANTS
        ],
        write_template="Fact: {subject} {phrase} {target_new}.",
        read_template="prompt.format(subject)",
        enabled_modules=["HTWR"],
        disabled_modules=["AttnNativeBank", "SCAR", "CAA", "RSM"],
        extra={
            "tier": "T0",
            "phase": "alpha",
            "eta": eta,
            "hook_point": args.hook_point,
            "bank_size": args.bank_size,
            "n_prompts": n_prompts,
            "inject_layers": args.inject_layers or "all",
            "retrievers": "OracleRetriever + RandomRetriever",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Exp12 HTWR — Phase α / Tier 0")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--counterfact", default=str(ROOT / "experiments/datasets/counterfact_1k.jsonl"))
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--eta-grid", default="",
                   help="comma-list. If set, sweeps eta and creates subdirs eta_<v>.")
    p.add_argument("--hook-point", default="block_output",
                   choices=["block_output", "pre_block_input", "mlp_mid"])
    p.add_argument("--inject-layers", default="",
                   help="comma-list of layer indices. Empty = all layers.")
    p.add_argument("--bank-size", type=int, default=64)
    p.add_argument("--n-prompts", type=int, default=100)
    args = p.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    cf_path = Path(args.counterfact)
    inject_layers = (
        tuple(int(x) for x in args.inject_layers.split(",") if x)
        if args.inject_layers else None
    )

    _log(f"loading {args.model} dtype={args.dtype} device={args.device}")
    tok, model = load_model(args.model, device=args.device, dtype=args.dtype)

    # Build one big bank with eta=0 injector (capture only — eta doesn't affect capture).
    cap_cfg = HTWRConfig(eta=0.0, hook_point=args.hook_point)
    cap_inj = HTWRInjector(model, cap_cfg)
    rows = _eligible_rows(cf_path, tok, args.n_prompts)
    bank = _build_bank(cap_inj, tok, rows, args.device)
    _log(f"bank.memories shape = {tuple(bank.memories.shape)}  hook={args.hook_point}")

    eta_list = (
        [float(x) for x in args.eta_grid.split(",") if x]
        if args.eta_grid else [args.eta]
    )

    summaries = []
    for eta in eta_list:
        tag = f"eta_{eta:.3f}".replace(".", "_")
        cfg_out = out_root / tag if len(eta_list) > 1 else out_root
        cfg_out.mkdir(parents=True, exist_ok=True)
        _write_manifest(cfg_out, args=args, cf_path=cf_path, eta=eta, n_prompts=len(rows))
        _log(f"running T0 eta={eta} hook={args.hook_point}")
        res = _run_tier0(
            model=model, tok=tok, rows=rows, full_bank=bank,
            device=args.device, out_dir=cfg_out, eta=eta,
            hook_point=args.hook_point, bank_size=args.bank_size,
            seeds=seeds, inject_layers=inject_layers,
        )
        summary = aggregate(
            res, experiment=f"{EXPERIMENT}_t0_{tag}",
            model=args.model, dataset=cf_path.name, out_dir=cfg_out,
        )
        summaries.append({"eta": eta, "tag": tag, "summary": str(summary)})
    (out_root / "phase_alpha_index.json").write_text(json.dumps(summaries, indent=2))
    _log("Phase α / Tier 0 complete.")


if __name__ == "__main__":
    main()
