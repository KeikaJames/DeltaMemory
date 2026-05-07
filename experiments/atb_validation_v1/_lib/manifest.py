"""Manifest writer for ATB validation v1 experiments.

Stamps a YAML manifest with: experiment id, code commit SHA, dataset SHA1,
model + dtype, attention_impl, seeds, variant list, write/read templates,
metric definitions, enabled/disabled modules, host/GPU info, timestamp.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_sha(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return bool(out.decode().strip())
    except Exception:
        return False


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device_count"] = int(torch.cuda.device_count())
            info["device_name_0"] = torch.cuda.get_device_name(0)
            info["torch_version"] = torch.__version__
            info["cuda_version"] = torch.version.cuda
        info["mps_available"] = bool(getattr(torch.backends, "mps",
                                              None) and
                                      torch.backends.mps.is_available())
    except Exception:
        pass
    return info


def _to_yaml(d: dict, indent: int = 0) -> str:
    """Tiny YAML dumper (no PyYAML dependency)."""
    out = []
    pad = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            out.append(f"{pad}{k}:")
            out.append(_to_yaml(v, indent + 1))
        elif isinstance(v, list):
            out.append(f"{pad}{k}:")
            for item in v:
                if isinstance(item, dict):
                    items = list(item.items())
                    if not items:
                        out.append(f"{pad}- {{}}")
                        continue
                    first_k, first_v = items[0]
                    out.append(f"{pad}- {first_k}: {_yaml_scalar(first_v)}")
                    for kk, vv in items[1:]:
                        if isinstance(vv, (dict, list)):
                            out.append(f"{pad}  {kk}:")
                            if isinstance(vv, dict):
                                out.append(_to_yaml(vv, indent + 2))
                            else:
                                for x in vv:
                                    out.append(f"{pad}    - {_yaml_scalar(x)}")
                        else:
                            out.append(f"{pad}  {kk}: {_yaml_scalar(vv)}")
                else:
                    out.append(f"{pad}- {_yaml_scalar(item)}")
        else:
            out.append(f"{pad}{k}: {_yaml_scalar(v)}")
    return "\n".join(out)


def _yaml_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    if any(c in s for c in ":#\n") or s.startswith(" ") or s.endswith(" "):
        return json.dumps(s)
    return s


def write_manifest(
    out_dir: Path,
    experiment: str,
    *,
    repo_root: Path,
    dataset_path: Path | None,
    dataset_sha1: str | None,
    model: str,
    dtype: str,
    attention_impl: str,
    seeds: list[int],
    variants: list[dict],
    write_template: str,
    read_template: str,
    enabled_modules: list[str],
    disabled_modules: list[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``out_dir/manifest.yaml`` and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": experiment,
        "schema_version": "atb_validation_v1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "code": {
            "repo": "RCV-HC",
            "git_sha": _git_sha(repo_root),
            "git_dirty": _git_dirty(repo_root),
        },
        "model": model,
        "dtype": dtype,
        "attention_impl": attention_impl,
        "dataset": {
            "path": str(dataset_path) if dataset_path else None,
            "sha1": dataset_sha1,
        },
        "seeds": seeds,
        "variants": variants,
        "templates": {
            "write": write_template,
            "read": read_template,
        },
        "modules": {
            "enabled": enabled_modules,
            "disabled": disabled_modules,
        },
        "metric_definitions": {
            "recall_at_1": "argmax of next-token logits at end of read prompt equals first token of target_new",
            "margin": "sum_logp(target_new | prompt) - sum_logp(target_true | prompt) over all target tokens",
            "target_rank": "0-indexed rank of first target_new token in next-token distribution",
            "js_drift": "mean symmetric Jensen-Shannon divergence (nats) over last-8 logits across 100 fixed neutral prompts",
            "kl_drift": "mean KL(p_baseline || p_patched) (nats) over last-8 logits across 100 fixed neutral prompts",
            "bank_attention_mass": "sum of merged-softmax weights over bank columns, mean across (B,H,T,layers)",
            "max_bank_prob": "max merged-softmax weight on any bank column, mean across (B,H,T,layers)",
        },
        "gpu": _gpu_info(),
    }
    if extra:
        manifest["extra"] = extra
    p = out_dir / "manifest.yaml"
    p.write_text(_to_yaml(manifest) + "\n")
    return p
