from __future__ import annotations

import warnings as _warnings
_warnings.filterwarnings("ignore", category=DeprecationWarning)
import pytest
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

from pathlib import Path

from deltamemory.engine.prototype import (
    PrototypeRunConfig,
    run_attention_memory_prototype,
    write_prototype_report,
)


def test_gemma4_prototype_runner_mock(tmp_path):
    input_path = tmp_path / "demo.txt"
    input_path.write_text(
        "The secret code for unit XJQ-482 is tulip-91. "
        "The secret code for unit XJQ-483 is tulip-19. "
        "The unit XJQ-482 was later selected for verification.",
        encoding="utf-8",
    )
    cfg = PrototypeRunConfig(
        model="mock-gemma",
        device="cpu",
        dtype="float32",
        input_path=str(input_path),
        store_path=str(tmp_path / "store"),
        report_dir=str(tmp_path / "report"),
        block_size=8,
        memory_dim=32,
        top_k=2,
    )
    summary = run_attention_memory_prototype(cfg)
    assert summary["path"] == "external_attention_memory_to_gemma_qkv"
    assert summary["prompt_insertion_used"] is False
    assert summary["ingest"]["trainable_base_params"] == 0
    assert summary["diagnosis"]["delta_qv_q_nonzero"] is True
    assert summary["diagnosis"]["delta_qv_v_nonzero"] is True
    paths = write_prototype_report(summary, tmp_path / "report")
    assert Path(paths["report"]).exists()
    assert "Delta Memory Gemma4 Layerwise Injection Prototype" in Path(paths["report"]).read_text(encoding="utf-8")
