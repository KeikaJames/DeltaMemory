#!/usr/bin/env python3
"""Compare static Gaussian, sigma-radius, and logit-lens-proxy layer rules.

The script is intentionally dependency-light: it can consume existing U-LOPI
profile.json files, or future N=100 profiles with the same schema.
"""
from __future__ import annotations

import argparse, json, math, platform, subprocess
from pathlib import Path


def _env(dtype: str, device: str) -> dict:
    try:
        import torch
        torch_v = torch.__version__
    except Exception as exc:  # pragma: no cover
        torch_v = repr(exc)
    try:
        import transformers
        hf_v = transformers.__version__
    except Exception as exc:  # pragma: no cover
        hf_v = repr(exc)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    return {"torch": torch_v, "transformers": hf_v, "commit": commit, "dtype": dtype, "device": device, "python": platform.python_version()}


def analyze_profile(path: Path) -> dict:
    p = json.loads(path.read_text())
    sig = [float(x) for x in p["sigma_base"]]
    mu = [float(x) for x in p["mu_base"]]
    L = len(sig)
    static_mu = L / 2.0
    static_sigma = L / 6.0
    sigma_argmax = max(range(L), key=lambda i: sig[i])
    # Residual spectral-radius proxy: largest coefficient of variation-normalized sigma.
    mean_sig = sum(sig) / L
    spectral_scores = [s / mean_sig for s in sig]
    spectral_argmax = max(range(L), key=lambda i: spectral_scores[i])
    # Logit-lens information-gain proxy when logits are unavailable: peak positive
    # change in residual norm, i.e. where block output changes most from previous.
    gains = [0.0] + [max(0.0, mu[i] - mu[i - 1]) for i in range(1, L)]
    logit_proxy_argmax = max(range(L), key=lambda i: gains[i])
    return {
        "profile": str(path), "L": L, "profile_mu_arch": int(p.get("mu_arch", sigma_argmax)),
        "static_gaussian": {"mu": static_mu, "sigma": static_sigma, "peak_layer": round(static_mu)},
        "sigma_radius_rule": {"argmax_layer": sigma_argmax, "score": sig[sigma_argmax]},
        "residual_spectral_proxy": {"argmax_layer": spectral_argmax, "score": spectral_scores[spectral_argmax]},
        "logit_lens_gain_proxy": {"argmax_layer": logit_proxy_argmax, "score": gains[logit_proxy_argmax]},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="*", default=["experiments/W_T3_6_ulopi_profiler/qwen05/profile.json", "experiments/W_T3_6_ulopi_profiler/qwen15/profile.json"])
    ap.add_argument("--out", default="experiments/A3_layer_weighting/raw_cells.json")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = [analyze_profile(Path(p)) for p in args.profiles if Path(p).exists()]
    out.write_text(json.dumps({"env": _env(args.dtype, args.device), "rows": rows}, indent=2), encoding="utf-8")
    (out.parent / "env.json").write_text(json.dumps(_env(args.dtype, args.device), indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main()
