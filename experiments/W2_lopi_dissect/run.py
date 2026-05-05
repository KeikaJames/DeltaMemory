"""W.2 LOPI 3-component dissection — CLI runner.

Grid: 3 models × 7 α × 5 arms × 3 seeds × 30 prompts = 9450 forward passes.

Usage
-----
    python experiments/W2_lopi_dissect/run.py \\
        --out experiments/W2_lopi_dissect/cells.jsonl \\
        --device mps --dtype bfloat16

Supports incremental checkpointing: if --out already exists, cells already
completed are skipped (keyed on model × arm × alpha × seed × prompt_id).

Arms
----
A0 = LOPI OFF (bank injection with raw M_V, no LOPI modulation)
A1 = M⊥ only  (orthogonal=True, gaussian=False, derivative=False)
A2 = M⊥ + Gauss (orthogonal=True, gaussian=True, derivative=False)
A3 = Full LOPI  (orthogonal=True, gaussian=True, derivative=True)
A4 = Gauss + γ  (orthogonal=False, gaussian=True, derivative=True)
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deltamemory.diagnostics import DiagnosticRecorder, _RECORDER  # noqa: F401
import deltamemory.diagnostics as _diag_mod
from deltamemory.memory.attn_native_bank import (
    AttnNativeBank,
    AttnNativePatcher,
    fresh_bank,
    write_fact,
)
from deltamemory.memory.lopi import LOPIConfig, LOPIState, apply_lopi
from deltamemory.memory.mhc_shield import shield_attention_weights

# ---------------------------------------------------------------------------
# Constants

MODELS = [
    "gpt2-medium",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",   # substituted for meta-llama/Llama-3.2-1B (gated, no token)
]

ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

SEEDS = [0, 1, 2]

ARMS: dict[str, dict] = {
    "A0": {"enabled": False,
            "orthogonal": False, "gaussian": False, "derivative": False},
    "A1": {"enabled": True,
            "orthogonal": True,  "gaussian": False, "derivative": False},
    "A2": {"enabled": True,
            "orthogonal": True,  "gaussian": True,  "derivative": False},
    "A3": {"enabled": True,
            "orthogonal": True,  "gaussian": True,  "derivative": True},
    "A4": {"enabled": True,
            "orthogonal": False, "gaussian": True,  "derivative": True},
}

# Neutral write-fact (same as R-3; unrelated to gold prompts topic).
WRITE_PROMPT = "Fact: The Sun is a star at the centre of the Solar System."
WRITE_FACT_ID = "neutral_anchor"
WRITE_ADDRESS = "the Sun"

PROMPTS_PATH = ROOT / "experiments" / "datasets" / "gold_30prompts.jsonl"

# ---------------------------------------------------------------------------
# Diagnostic recorder extension for GPT-2 (adds "transformer.h" path)


class W2DiagnosticRecorder(DiagnosticRecorder):
    """DiagnosticRecorder with extended decoder-layer discovery (adds GPT-2)."""

    def _find_decoder_layers(self) -> list:
        try:
            return super()._find_decoder_layers()
        except RuntimeError:
            pass
        # GPT-2 path
        model = self._model
        for path in ("transformer.h", "model.transformer.h"):
            obj: Any = model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok and hasattr(obj, "__len__") and len(obj) > 0:
                return list(obj)
        raise RuntimeError(
            "W2DiagnosticRecorder: cannot locate decoder layers "
            f"on {type(model).__name__}"
        )


# ---------------------------------------------------------------------------
# Helpers


def load_prompts() -> list[dict]:
    prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def load_model(model_name: str, device: str, dtype: torch.dtype):
    """Load (tokenizer, model) for the given model family."""
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


def is_rope_model(model_name: str) -> bool:
    """Return True for models supported by AttnNativePatcher (RoPE-based)."""
    return model_name.startswith("Qwen/") or "llama" in model_name.lower()


def compute_baseline_nlls(model, tok, prompts: list[dict], device: str) -> list[float]:
    """NLL for each prompt with no bank (pure model forward)."""
    nlls = []
    with torch.no_grad():
        for p in prompts:
            text = p["text"]
            enc = tok(text, return_tensors="pt", add_special_tokens=True,
                       truncation=True, max_length=256)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=am, use_cache=False)
            targets = ids[0, 1:]
            logp = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
            nll = -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1).mean().item()
            nlls.append(float(nll))
    return nlls


# ---------------------------------------------------------------------------
# RoPE-model path (Qwen / Llama) — uses AttnNativePatcher + DiagnosticRecorder


def _run_rope_cell(
    model, tok, patcher: AttnNativePatcher,
    bank: AttnNativeBank, base_bank: AttnNativeBank,
    prompts: list[dict], base_nlls: list[float],
    arm: str, alpha: float, seed: int, device: str,
    model_name: str = "unknown",
) -> list[dict]:
    """Run one (arm, alpha, seed) cell on a RoPE model. Returns list of per-prompt dicts."""
    torch.manual_seed(seed)

    # Configure LOPI on the bank.
    arm_cfg = ARMS[arm]
    bank.lopi_cfg = LOPIConfig(**arm_cfg)
    bank.lopi_state = LOPIState(num_layers=bank.num_layers)
    bank.mhc_shield = True

    cells = []
    with W2DiagnosticRecorder(
        model, patcher, lopi_state=bank.lopi_state, enabled=True
    ) as rec:
        for pi, prompt in enumerate(prompts):
            rec._records.clear()
            rec._current_step = -1

            text = prompt["text"]
            enc = tok(text, return_tensors="pt", add_special_tokens=True,
                       truncation=True, max_length=256)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)

            bank.lopi_state.reset()
            with patcher.patched(), patcher.injecting(bank, alpha=alpha), torch.no_grad():
                out = model(input_ids=ids, attention_mask=am, use_cache=False)

            # NLL
            targets = ids[0, 1:]
            logp = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
            inj_nll = float(
                -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1).mean().item()
            )
            drift = inj_nll - base_nlls[pi]

            # Collect diagnostics from recorder
            diag = _extract_diag(rec)

            cells.append({
                "model": model_name,
                "arm": arm,
                "alpha": alpha,
                "seed": seed,
                "prompt_id": prompt["id"],
                "inj_nll": inj_nll,
                "base_nll": base_nlls[pi],
                "drift": drift,
                **diag,
            })
    return cells


def _extract_diag(rec: DiagnosticRecorder) -> dict:
    """Extract aggregated diagnostics from a DiagnosticRecorder."""
    import numpy as np

    df = rec.to_pandas()
    result: dict[str, Any] = {}

    if df.empty:
        return result

    # lopi_gamma_t stats
    gt = df[df["signal_name"] == "lopi_gamma_t"]["value"].values
    if len(gt) > 0:
        result["lopi_gamma_t_mean"] = float(np.mean(gt))
        result["lopi_gamma_t_p50"] = float(np.percentile(gt, 50))
        result["lopi_gamma_t_p99"] = float(np.percentile(gt, 99))

    # lopi_w_ell per layer — argmax
    wl = df[df["signal_name"] == "lopi_w_ell"]
    if not wl.empty:
        layer_max = wl.groupby("layer")["value"].mean()
        result["lopi_w_ell_argmax"] = int(layer_max.idxmax())
        result["lopi_w_ell_max"] = float(layer_max.max())
        result["lopi_w_ell_per_layer"] = layer_max.to_dict()

    # m_perp_energy_ratio per layer mean
    mp = df[df["signal_name"] == "m_perp_energy_ratio"]
    if not mp.empty:
        result["m_perp_energy_ratio_mean"] = float(mp["value"].mean())
        result["m_perp_energy_ratio_per_layer"] = (
            mp.groupby("layer")["value"].mean().to_dict()
        )

    # residual_norm_p50 per layer
    rn = df[df["signal_name"] == "residual_norm"]
    if not rn.empty:
        result["residual_norm_p50"] = float(rn["value"].median())
        result["residual_norm_per_layer_p50"] = (
            rn.groupby("layer")["value"].median().to_dict()
        )

    return result


# ---------------------------------------------------------------------------
# GPT-2 path — legacy hook-based forward, manual LOPI application


def _gpt2_run_cell(
    model, tok,
    prompts: list[dict], base_nlls: list[float],
    arm: str, alpha: float, seed: int, device: str,
) -> list[dict]:
    """Run one cell for GPT-2 using hook-based bank injection."""
    torch.manual_seed(seed)

    n_layers = model.config.n_layer
    arm_cfg = ARMS[arm]
    lopi_cfg = LOPIConfig(**arm_cfg)
    lopi_state = LOPIState(num_layers=n_layers)

    # Build a bank the GPT-2 way: one K/V dict per layer.
    bank = _gpt2_write_fact(model, tok, WRITE_PROMPT, n_layers, device)

    # Use W2DiagnosticRecorder to collect LOPI diagnostics even for GPT-2.
    # The residual-norm hooks will find GPT-2 layers via the extended path.
    # Pass lopi_state so record_lopi_gamma_w can write gamma_t signals.
    rec = W2DiagnosticRecorder(model, None, lopi_state=lopi_state, enabled=True)

    cells = []

    with rec:
        for pi, prompt in enumerate(prompts):
            rec._records.clear()
            rec._current_step = -1
            lopi_state.reset()

            text = prompt["text"]
            enc = tok(text, return_tensors="pt", add_special_tokens=True,
                       truncation=True, max_length=256)
            ids = enc["input_ids"].to(device)
            am = enc["attention_mask"].to(device)

            inj_nll = _gpt2_seq_nll(
                model, tok, ids, am, bank, alpha, n_layers, device,
                lopi_cfg=lopi_cfg, lopi_state=lopi_state, recorder=rec,
            )
            drift = inj_nll - base_nlls[pi]
            diag = _extract_diag(rec)

            cells.append({
                "model": "gpt2-medium",
                "arm": arm,
                "alpha": alpha,
                "seed": seed,
                "prompt_id": prompt["id"],
                "inj_nll": float(inj_nll),
                "base_nll": base_nlls[pi],
                "drift": drift,
                **diag,
            })

    return cells


def _gpt2_write_fact(model, tok, write_prompt: str, n_layers: int, device: str):
    """Capture K/V for GPT-2 using the legacy approach from R-3."""
    enc = tok(write_prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"].to(device)
    captures: list[dict] = [{}] * n_layers
    layer_caps: list[dict] = [{} for _ in range(n_layers)]
    handles = []

    def make_cap_hook(li):
        def hook(module, inp, out):
            x = inp[0]
            B, T, C = x.shape
            qkv = module.c_attn(x)
            split = module.split_size
            q, k, v = qkv.split(split, dim=-1)
            n_head = module.num_heads
            hd = C // n_head
            k_r = k.view(B, T, n_head, hd).transpose(1, 2)  # [B, H, T, hd]
            v_r = v.view(B, T, n_head, hd).transpose(1, 2)
            # Capture last token
            layer_caps[li]["K"] = k_r[:, :, -1:, :].detach().mean(dim=2)  # [B, H, hd]
            layer_caps[li]["V"] = v_r[:, :, -1:, :].detach().mean(dim=2)
        return hook

    for li in range(n_layers):
        m = model.transformer.h[li].attn
        h = m.register_forward_hook(make_cap_hook(li))
        handles.append(h)

    with torch.no_grad():
        model(input_ids=ids, use_cache=False)

    for h in handles:
        h.remove()

    return layer_caps


@torch.no_grad()
def _gpt2_seq_nll(
    model, tok, ids, am, bank, alpha, n_layers, device,
    lopi_cfg, lopi_state, recorder,
) -> float:
    """GPT-2 bank-injected NLL with mHC shield + LOPI."""
    B, T = ids.shape
    handles = []

    def make_hook(li):
        mk = bank[li]["K"].to(device)  # [B, H, hd]
        mv = bank[li]["V"].to(device)

        def hook(module, inp, out):
            if alpha == 0.0:
                return out

            x = inp[0]
            Bx, Tx, C = x.shape
            n_head = module.num_heads
            hd = C // n_head

            qkv = module.c_attn(x)
            split = module.split_size
            q, k, v = qkv.split(split, dim=-1)

            q_r = q.view(Bx, Tx, n_head, hd).transpose(1, 2)  # [B, H, T, hd]
            k_r = k.view(Bx, Tx, n_head, hd).transpose(1, 2)
            v_r = v.view(Bx, Tx, n_head, hd).transpose(1, 2)

            mk_r = mk.view(1, n_head, 1, hd).expand(Bx, -1, 1, -1)
            mv_r = mv.view(1, n_head, 1, hd).expand(Bx, -1, 1, -1)

            k_cat = torch.cat([k_r, mk_r], dim=-2)  # [B, H, T+1, hd]
            scale = hd ** -0.5
            scores = torch.matmul(q_r, k_cat.transpose(-2, -1)) * scale

            # Causal mask on sequence columns only
            mask = torch.triu(
                torch.ones(Tx, Tx, dtype=torch.bool, device=device), diagonal=1
            )
            scores[..., :Tx].masked_fill_(mask, float("-inf"))

            w = torch.softmax(scores.float(), dim=-1).to(q_r.dtype)

            # mHC shield on bank columns
            w = shield_attention_weights(w, bank_size=1, enabled=True)

            out_orig = torch.matmul(w[..., :Tx], v_r)
            out_bank_raw = torch.matmul(w[..., Tx:], alpha * mv_r)

            # LOPI (commit step once per layer-0 encounter in this forward)
            if li == 0:
                lopi_state.commit_step()
            out_bank = apply_lopi(
                out_bank_native=out_bank_raw,
                v_ctx_readout=out_orig,
                q_post=q_r,
                layer_idx=li,
                state=lopi_state,
                cfg=lopi_cfg,
            )
            lopi_state.pending_residual_norms[li] = float(
                torch.linalg.vector_norm(out_orig.float(), ord=2, dim=-1).mean().item()
            )

            attn_out = (out_orig + out_bank).transpose(1, 2).contiguous().view(Bx, Tx, C)
            attn_out = module.c_proj(attn_out)
            attn_out = module.resid_dropout(attn_out)
            return (attn_out, None)

        return hook

    for li in range(n_layers):
        h = model.transformer.h[li].attn.register_forward_hook(make_hook(li))
        handles.append(h)

    out = model(input_ids=ids, attention_mask=am, use_cache=False)

    for h in handles:
        h.remove()

    targets = ids[0, 1:]
    logp = F.log_softmax(out.logits[0, :-1].float(), dim=-1)
    nll = -logp.gather(1, targets.unsqueeze(-1)).squeeze(-1).mean().item()
    return float(nll)


# ---------------------------------------------------------------------------
# Checkpoint helpers


def load_checkpoint(path: Path) -> set[str]:
    """Return set of already-completed cell keys."""
    done: set[str] = set()
    if not path.exists():
        return done
    open_fn = gzip.open if str(path).endswith(".gz") else open
    try:
        with open_fn(path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    key = f"{c['model']}|{c['arm']}|{c['alpha']}|{c['seed']}|{c['prompt_id']}"
                    done.add(key)
    except Exception:
        pass
    return done


def append_cell(path: Path, cell: dict) -> None:
    open_fn = gzip.open if str(path).endswith(".gz") else open
    mode = "at"
    with open_fn(path, mode) as f:
        f.write(json.dumps(cell) + "\n")


# ---------------------------------------------------------------------------
# Main


def main():
    ap = argparse.ArgumentParser(description="W.2 LOPI dissect runner")
    ap.add_argument("--out", default="/tmp/deltamemory/W2_lopi_dissect/cells.jsonl")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 1 model, 1 alpha, 1 seed, 3 prompts.")
    ap.add_argument("--n-prompts", type=int, default=30,
                    help="Cap number of prompts (for partial runs).")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = args.device

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        models = [args.models[0]]
        alphas = [args.alphas[0]]
        seeds = [0]
        n_prompts = 3
        arms = list(ARMS.keys())
    else:
        models = args.models
        alphas = args.alphas
        seeds = args.seeds
        n_prompts = args.n_prompts
        arms = args.arms

    prompts_all = load_prompts()[:n_prompts]
    done = load_checkpoint(out_path)

    total = len(models) * len(arms) * len(alphas) * len(seeds) * len(prompts_all)
    done_count = len(done)
    print(f"[W2] total cells = {total}, already done = {done_count}", flush=True)

    passes_run = 0
    t0 = time.time()

    for model_name in models:
        try:
            tok, model = load_model(model_name, device, dtype)
        except Exception as exc:
            print(f"[skip] {model_name}: {exc!r}", flush=True)
            traceback.print_exc()
            continue

        use_rope = is_rope_model(model_name)

        # Compute baseline NLLs (no bank)
        print(f"[W2] computing baselines for {model_name}...", flush=True)
        base_nlls = compute_baseline_nlls(model, tok, prompts_all, device)

        if use_rope:
            patcher = AttnNativePatcher(model)
            # Build bank (attach LOPI profile for auto mode)
            bank = fresh_bank(model)
            bank.mhc_shield = True
            bank.value_scale_mode = "auto_rms_cap"
            bank.value_target_rms = 0.5
            write_fact(patcher, bank, tok,
                       write_prompt=WRITE_PROMPT,
                       fact_id=WRITE_FACT_ID,
                       address=WRITE_ADDRESS)
            try:
                bank.attach_lopi_profile(model, tok)
                print(f"[W2] LOPI profile attached for {model_name}: "
                      f"mu_arch={bank.lopi_state.profile.mu_arch}", flush=True)
            except Exception as exc:
                print(f"[W2] LOPI profile attach failed: {exc!r} — using static mode", flush=True)

        for arm in arms:
            for alpha in alphas:
                for seed in seeds:
                    # Check if all prompts in this cell are done
                    all_done = all(
                        f"{model_name}|{arm}|{alpha}|{seed}|{p['id']}" in done
                        for p in prompts_all
                    )
                    if all_done:
                        passes_run += len(prompts_all)
                        continue

                    t_cell = time.time()
                    try:
                        if use_rope:
                            cell_rows = _run_rope_cell(
                                model, tok, patcher, bank, bank,
                                prompts_all, base_nlls,
                                arm, alpha, seed, device,
                                model_name=model_name,
                            )
                        else:
                            cell_rows = _gpt2_run_cell(
                                model, tok, prompts_all, base_nlls,
                                arm, alpha, seed, device,
                            )

                        for row in cell_rows:
                            key = (f"{row['model']}|{row['arm']}|"
                                   f"{row['alpha']}|{row['seed']}|{row['prompt_id']}")
                            if key not in done:
                                append_cell(out_path, row)
                                done.add(key)
                                passes_run += 1

                    except Exception as exc:
                        print(f"[ERROR] {model_name} {arm} α={alpha} seed={seed}: "
                              f"{exc!r}", flush=True)
                        traceback.print_exc()
                        continue

                    elapsed = time.time() - t_cell
                    total_elapsed = time.time() - t0
                    remaining = total - passes_run
                    eta = (total_elapsed / max(passes_run, 1)) * remaining if passes_run > 0 else 0
                    print(
                        f"[{passes_run:>5}/{total}] {model_name[:12]:12s} "
                        f"{arm} α={alpha:<5.1f} seed={seed} "
                        f"cell={elapsed:.1f}s eta={eta/3600:.1f}h",
                        flush=True,
                    )

        # Clean up
        del model, tok
        if device == "mps":
            torch.mps.empty_cache()

    total_elapsed = time.time() - t0
    print(
        f"\n[W2] DONE: {passes_run}/{total} passes in {total_elapsed/3600:.2f}h",
        flush=True,
    )
    print(f"[W2] Output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
