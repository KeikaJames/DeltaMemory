"""deltamemory.utils.run_env — mandatory env.json writer for all runners.

Every experiment runner **must** call :func:`write_env_json` at startup so
that every ``runs/*/`` directory has a verifiable ``env.json`` capturing the
exact reproducibility context (commit, dirty state, model path, GPU, dtype,
library versions).

This module is a thin, discoverable re-export of the canonical implementation
in ``tools/env_writer.py``.  New code should import from here; ``tools/``
remains for legacy compatibility.

Usage::

    from deltamemory.utils.run_env import write_env_json

    write_env_json(
        out_dir=Path("runs/my_experiment"),
        prereg_version="X7NL.v1",
        dataset_sha1=sha1_of("experiments/datasets/counterfact_60.jsonl"),
        device="cuda",
        dtype="bfloat16",
        extra={"model_path": "/path/to/model", "alpha": 1.0},
    )

The ``env.json`` file written by this function contains:

    commit           — full git SHA of HEAD
    dirty            — True if working tree has uncommitted changes
    dirty_diff_sha1  — sha1 of ``git diff`` output (reproducibility anchor)
    prereg_version   — pre-registration identifier (e.g. "X7NL.v1")
    dataset_sha1     — sha1 of dataset file(s)
    torch            — torch.__version__
    transformers     — transformers.__version__
    python           — Python version
    device           — device string ("cuda", "mps", "cpu")
    dtype            — dtype string ("bfloat16", "float32", …)
    started_at       — UTC ISO-8601 timestamp
    host             — hostname
    gpu_name         — GPU device name (when CUDA is available)
    cli_argv         — sys.argv at call time
"""
from __future__ import annotations

# Re-export everything from the canonical implementation.
# Any future extensions should be added to tools/env_writer.py and re-exported
# here, to keep a single source of truth.
from tools.env_writer import build_env, sha1_of, write_env_json  # noqa: F401

__all__ = ["build_env", "sha1_of", "write_env_json"]
