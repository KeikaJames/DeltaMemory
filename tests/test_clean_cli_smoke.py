from __future__ import annotations

import json
from pathlib import Path

from rcvhc.cli import main


def test_clean_cli_ingest_ask_inspect(tmp_path, capsys):
    input_path = tmp_path / "demo.txt"
    input_path.write_text(
        "The secret code for unit XJQ-482 is tulip-91. "
        "The unit XJQ-482 was later selected for verification.",
        encoding="utf-8",
    )
    store = tmp_path / "store"
    assert main(["ingest", "--model", "mock-gemma", "--input", str(input_path), "--store", str(store), "--block-size", "8", "--memory-dim", "32"]) == 0
    ingest_out = json.loads(capsys.readouterr().out)
    assert ingest_out["memory_blocks"] > 0

    assert main([
        "ask",
        "--model",
        "mock-gemma",
        "--store",
        str(store),
        "--question",
        "What is the secret code for unit XJQ-482?",
        "--answer",
        "tulip-91",
        "--mode",
        "delta_qv",
    ]) == 0
    ask_out = json.loads(capsys.readouterr().out)
    assert ask_out["source_text_used_in_prompt"] is False
    assert "delta_qv_force_gate" in ask_out["comparisons"]

    assert main(["inspect", "--store", str(store)]) == 0
    inspect_out = json.loads(capsys.readouterr().out)
    assert inspect_out["memory_count"] > 0
