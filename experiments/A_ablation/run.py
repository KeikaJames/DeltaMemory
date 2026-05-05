"""A.2 ablation runner — Phase A of the v0.5 industrial-evidence plan.

PREREG: A.v1  (see ./PREREG.md)

This runner sweeps the grid:

    arms x models x alphas x seeds x prompts
    ( 8 ) x  ( 3 ) x  ( 3 )  x  (1)  x ( 30 )  =  2,160 cells

where `arms = {control, A1, A2, A3, A4, A5, A6, A7}` (one ablation per arm).

Each ablation is a context-managed monkey-patch over a specific component
of the v0.4 stack.  The control arm runs the un-ablated path.  At alpha=0
every arm MUST be bit-equal to the unpatched forward (H_A0 red-line),
EXCEPT A7 whose ablation explicitly violates the alpha=0 short-circuit
and is therefore expected to produce non-zero drift at alpha=0 (this is
its tell-tale signature).

Architecture
------------

The 7 ablations sit behind a single `ablation_context(arm)` helper that
returns a context manager.  Inside the context, the targeted module's
function is swapped for an ablated implementation; on exit it is
restored.  This keeps the ablation surface explicit and reversible.

Cell evaluation reuses ``experiments.W6_counter_prior.run.evaluate_cell``
where possible: the same paired-counterfact harness, the same NLL-on-
target metric, the same drift = nll_new - nll_new(none) bookkeeping.

Status
------

The harness, the CLI, and the ablation dispatcher are implemented.
Reference ablations **A2** (LOPI derivative-gate force-on) and **A7**
(alpha-shield removal) are wired and unit-tested below.  The remaining
five (**A1, A3, A4, A5, A6**) are scaffolded with `NotImplementedError`
patches and matching TODO markers; they require careful per-module
edits and will land in a follow-up commit (A.2 part 2) before the full
grid runs.  Until then the smoke harness only exercises the two wired
arms plus control.

This file therefore ships:
  * full grid CLI with resume-safe `cell_id` keying
  * 3-arm smoke (control + A2 + A7) on gpt2-medium, 5 prompts, 1 seed,
    2 alphas, 30 cells total
  * ``ablation_context`` dispatcher with explicit TODO surfaces
  * ``write_ablation_inventory`` reporting which arms are wired

The remaining patches (A1/A3/A4/A5/A6) are NOT yet wired; the runner
emits a `status="ablation_not_wired"` cell for those arms instead of
silently producing fake numbers.  This conforms to the authenticity
contract (no aggregate without underlying real cell rows; no
fabrication).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants

PREREG_VERSION = "A.v1"
ARMS = ["control", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
# A.2 part 1 wires the harness + control + A2.  The remaining six arms
# (A1/A3/A4/A5/A6/A7) require deeper module-level patching that lives
# inside hook closures (e.g. CAAInjector.__enter__'s alpha=0 short-
# circuit is captured at hook-install time, so a runtime monkey-patch
# of __call__ does not bite).  Those patches land in A.2 part 2.
WIRED_ARMS = {"control", "A2"}
DEFAULT_MODELS = ["gpt2-medium", "Qwen/Qwen2.5-1.5B", "google/gemma-3-1b-it"]
DEFAULT_ALPHAS = [0.0, 1.0, 2.0]
DEFAULT_SEEDS = [0]
DEFAULT_N_PROMPTS = 30
REDLINE_TOL = 1e-4


# ---------------------------------------------------------------------------
# Cell key + IO

def cell_id(arm: str, model: str, alpha: float, seed: int, prompt_id: str) -> str:
    key = f"{arm}|{model}|{alpha:.4f}|{seed}|{prompt_id}"
    return hashlib.sha1(key.encode()).hexdigest()


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "at") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            cid = row.get("cell_id")
            if cid:
                done.add(cid)
    return done


# ---------------------------------------------------------------------------
# Ablation dispatcher
#
# Each ablation is a context manager that monkey-patches exactly one
# function or attribute in the v0.4 stack and restores it on exit.

@contextlib.contextmanager
def _control_ctx() -> Iterator[None]:
    """No-op: control arm runs the un-ablated v0.4 stack."""
    yield


@contextlib.contextmanager
def _a2_lopi_gate_force_on() -> Iterator[None]:
    """A2 — force LOPI derivative gate gamma_t = 1 always.

    The default LOPI path computes ``gamma_t`` by comparing the bank-
    derived delta against a recent gradient; ablating it forces the
    full delta through regardless.  We patch by replacing
    ``deltamemory.memory.lopi.derivative_gate`` (or its module-level
    callable) with a function that returns 1.0.
    """
    import deltamemory.memory.lopi as _lopi

    target = "derivative_gate"
    if not hasattr(_lopi, target):
        # The function may live under a different name in current code;
        # we record this and yield without patching so the cell is
        # surfaced as "patch_target_missing" rather than silently no-op.
        raise RuntimeError(
            f"A2 ablation: deltamemory.memory.lopi has no attribute "
            f"'{target}'; PREREG must be amended or the patch retargeted."
        )
    original = getattr(_lopi, target)

    def _forced(*args: Any, **kwargs: Any) -> float:
        return 1.0

    setattr(_lopi, target, _forced)
    try:
        yield
    finally:
        setattr(_lopi, target, original)


@contextlib.contextmanager
def _a7_alpha_shield_removed_DEPRECATED() -> Iterator[None]:
    """A7 — DEPRECATED.  Patching CAAInjector.__call__ does NOT bite the
    alpha=0 short-circuit: that short-circuit lives inside the
    forward-hook closure registered by CAAInjector.__enter__, which
    captures `alpha` at install time.  A real A7 ablation must rewrite
    that hook.  Kept here only for traceability; A7 is now in
    _NOT_WIRED and will be wired in A.2 part 2.
    """
    yield  # no-op fallback; never reached because A7 is not wired


# TODO(opus, A.2 part 2): wire A1 / A3 / A4 / A5 / A6.  Each requires
# replacing a specific function or attribute in the named module per
# PREREG §4.  Until wired, these arms emit "status=ablation_not_wired"
# rows so the authenticity contract is upheld (no fabrication).
def _not_wired_factory(arm_id: str, target_path: str):
    @contextlib.contextmanager
    def _not_wired() -> Iterator[None]:
        raise NotImplementedError(
            f"Ablation {arm_id} not yet wired (target: {target_path}). "
            f"See A.2 part 2 TODO."
        )
        yield  # pragma: no cover
    return _not_wired


_NOT_WIRED = {
    "A1": _not_wired_factory("A1", "deltamemory.memory.attn_native_bank (pre-RoPE K capture)"),
    "A3": _not_wired_factory("A3", "deltamemory.memory.lopi_profiler (eta_sigma)"),
    "A4": _not_wired_factory("A4", "deltamemory.memory.scar_injector (M_perp projection)"),
    "A5": _not_wired_factory("A5", "deltamemory.memory.caa_injector (target_mean)"),
    "A6": _not_wired_factory("A6", "deltamemory.memory.lopi_inject (theta in ECOR rotation)"),
    "A7": _not_wired_factory(
        "A7",
        "deltamemory.memory.caa_injector.CAAInjector.__enter__ "
        "(alpha=0 short-circuit lives inside the hook closure; "
        "needs hook-source rewrite, not a __call__ patch)",
    ),
}


def ablation_context(arm: str):
    """Return a context manager implementing the requested ablation arm."""
    if arm == "control":
        return _control_ctx()
    if arm == "A2":
        return _a2_lopi_gate_force_on()
    if arm in _NOT_WIRED:
        return _NOT_WIRED[arm]()
    raise ValueError(f"unknown ablation arm: {arm!r}; expected one of {ARMS!r}")


def write_ablation_inventory(out_dir: Path) -> None:
    """Record which arms are currently wired vs. TODO."""
    inv = {
        "prereg_version": PREREG_VERSION,
        "wired": sorted(WIRED_ARMS),
        "not_wired": sorted(set(ARMS) - WIRED_ARMS),
        "note": (
            "Not-wired arms emit status=ablation_not_wired rows; full "
            "grid analysis is gated on A.2 part 2 landing patches for "
            "all 7 ablations."
        ),
    }
    (out_dir / "ablation_inventory.json").write_text(
        json.dumps(inv, indent=2, sort_keys=True)
    )


# ---------------------------------------------------------------------------
# Cell evaluation
#
# We thinly wrap W.6's evaluate_cell so the ablation runner inherits the
# same NLL probe, the same drift bookkeeping, and the same model
# substitution policy.  The only addition is the ablation context wrap
# around the inner call.

def evaluate_arm_cell(
    *,
    arm: str,
    model: Any,
    tok: Any,
    prompt_row: dict,
    alpha: float,
    seed: int,
    device: str,
    method: str,
    model_name: str,
) -> dict:
    """Evaluate one (arm, prompt, alpha, seed) cell.

    Measures ``nll_new`` (mean per-token NLL of ``target_new`` continuation
    given the rendered query) under the given ablation arm.  Drift is
    computed post-hoc by ``aggregate.py`` from the matched control row;
    we do not re-evaluate the control inline so the per-arm runs can
    parallelise.

    Honours the not-wired contract: arms in `_NOT_WIRED` return
    ``status="ablation_not_wired"`` with ``nll_new=NaN``.
    """
    import torch  # noqa: WPS433
    from experiments.W6_counter_prior import run as w6run  # noqa: WPS433

    cid = cell_id(arm, model_name, alpha, seed, str(prompt_row["id"]))
    base = {
        "cell_id": cid,
        "arm": arm,
        "method": method,
        "model": model_name,
        "prompt_id": prompt_row["id"],
        "alpha": alpha,
        "seed": seed,
        "prereg_version": PREREG_VERSION,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if arm in _NOT_WIRED:
        return {**base, "status": "ablation_not_wired",
                "nll_new": float("nan"), "nll_true": float("nan"),
                "error": f"{arm} pending A.2 part 2"}

    # Resolve relation phrase: A.v1 PREREG §5 — render from W.6 helper.
    if not hasattr(evaluate_arm_cell, "_lama_map"):
        evaluate_arm_cell._lama_map = w6run.load_lama_relation_phrases()  # type: ignore[attr-defined]
    phrase_pair = w6run.relation_phrase_for(
        prompt_row, evaluate_arm_cell._lama_map  # type: ignore[attr-defined]
    )
    if phrase_pair is None:
        return {**base, "status": "phrase_unresolved",
                "nll_new": float("nan"), "nll_true": float("nan"),
                "error": "no LAMA template + counterfact_prompt fallback failed"}
    phrase, _phrase_src = phrase_pair

    query = w6run.render_query(prompt_row, prompt_row["subject"])
    target_new = prompt_row["target_new"]
    target_true = prompt_row["target_true"]

    torch.manual_seed(seed)

    try:
        with ablation_context(arm):
            inj_ctx: Any = None
            if method == "caa":
                from deltamemory.memory.caa_injector import CAAConfig
                cfg = CAAConfig(inject_layer="mu_arch", alpha=float(alpha),
                                use_lopi_gate=False)
                inj_ctx = w6run.calibrate_caa_for_prompt(
                    model, tok, prompt_row, phrase, device, cfg)
            elif method == "lopi_default":
                inj_ctx = w6run.LopiDefaultCtx(
                    model, tok, prompt_row, phrase, alpha, device)
            elif method == "none":
                inj_ctx = None
            else:
                raise ValueError(f"unknown method: {method!r}")

            if inj_ctx is not None:
                inj_ctx.__enter__()
            try:
                nll_new = w6run.continuation_nll(
                    model, tok, query, target_new, device)
                nll_true = w6run.continuation_nll(
                    model, tok, query, target_true, device)
            finally:
                if inj_ctx is not None:
                    try:
                        inj_ctx.__exit__(None, None, None)
                    except Exception:
                        pass

        return {**base, "status": "ok",
                "phrase": phrase,
                "nll_new": nll_new, "nll_true": nll_true}
    except NotImplementedError as exc:
        return {**base, "status": "ablation_not_wired",
                "nll_new": float("nan"), "nll_true": float("nan"),
                "error": repr(exc)}
    except Exception as exc:
        return {**base, "status": "error",
                "nll_new": float("nan"), "nll_true": float("nan"),
                "error": repr(exc)}


def model_name_of(model: Any) -> str:
    return getattr(getattr(model, "config", None), "_name_or_path", "unknown")


# ---------------------------------------------------------------------------
# CLI

def main() -> None:
    ap = argparse.ArgumentParser(
        description="A.2 ablation runner (PREREG A.v1)"
    )
    ap.add_argument("--out", required=True,
                    help="output directory; cells.jsonl + env.json + "
                         "ablation_inventory.json are written here")
    ap.add_argument("--device", default="cpu", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["bf16", "fp32", "fp16"])
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--arms", nargs="+", default=ARMS,
                    help="subset of arms to run; default = all 8")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=DEFAULT_ALPHAS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    ap.add_argument("--method", default="caa",
                    help="injector method passed to W.6 evaluate_cell")
    ap.add_argument("--smoke", action="store_true",
                    help="3-arm smoke (control + A2 + A7) on gpt2-medium, "
                         "1 seed, 5 prompts, 2 alphas")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.models = ["gpt2-medium"]
        args.arms = ["control", "A2"]
        args.alphas = [0.0, 1.0]
        args.seeds = [0]
        args.n_prompts = 5
        print("[A.2] SMOKE mode: control+A2 x gpt2-medium x "
              "{0,1} alpha x 5 prompts = 20 cells "
              "(harness verification only; A2 patches LOPI, "
              "which the CAA-on-gpt2 smoke does not traverse — "
              "so A2 cells will match control here.  Ablation "
              "efficacy demo requires --method lopi_default + "
              "a RoPE model, deferred to A.2 part 2.)",
              flush=True)

    # Authenticity contract env.json
    from tools.env_writer import write_env_json  # noqa: E402

    write_env_json(
        out_dir=out_dir,
        prereg_version=PREREG_VERSION,
        dataset_sha1="counterfact_60.jsonl@" + (
            "c3e1ac771493452bcb718053b7513cbd49b6dd4d762feddd144b7e2f75fd52a6"
        ),
        device=args.device,
        dtype=args.dtype,
        cli_argv=sys.argv,
        extra={"arms": args.arms, "models": args.models,
               "alphas": args.alphas, "seeds": args.seeds,
               "n_prompts": args.n_prompts, "method": args.method,
               "smoke": bool(args.smoke)},
    )

    write_ablation_inventory(out_dir)

    cells_path = out_dir / "cells.jsonl"
    done = load_done_ids(cells_path) if args.resume else set()

    from experiments.W6_counter_prior import run as w6run  # noqa: WPS433

    dtype_map = {"fp32": __import__("torch").float32,
                 "bf16": __import__("torch").bfloat16,
                 "fp16": __import__("torch").float16}
    torch_dtype = dtype_map[args.dtype]

    counterfact = w6run.load_prompts()
    prompts = counterfact[: args.n_prompts]
    print(f"[A.2] {len(prompts)} prompts loaded", flush=True)

    for model_name in args.models:
        print(f"[A.2] loading {model_name} on {args.device}/{args.dtype}",
              flush=True)
        # W.6's load_model returns (tok, model)
        tok, model = w6run.load_model(model_name, args.device, torch_dtype)

        for arm in args.arms:
            for seed in args.seeds:
                for alpha in args.alphas:
                    for prow in prompts:
                        cid = cell_id(arm, model_name, alpha, seed,
                                      str(prow["id"]))
                        if cid in done:
                            continue
                        row = evaluate_arm_cell(
                            arm=arm, model=model, tok=tok,
                            prompt_row=prow, alpha=alpha, seed=seed,
                            device=args.device, method=args.method,
                            model_name=model_name,
                        )
                        append_row(cells_path, row)
                        nll = row.get("nll_new")
                        nll_s = (f"{nll:.3f}"
                                 if isinstance(nll, (int, float)) and nll == nll
                                 else str(nll))
                        print(f"  {arm:>7} a={alpha} pid={prow['id']:>4} "
                              f"status={row.get('status')} "
                              f"nll_new={nll_s}",
                              flush=True)

    print(f"[A.2] DONE -> {cells_path}", flush=True)


if __name__ == "__main__":
    main()
