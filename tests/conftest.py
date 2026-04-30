from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def demo_text() -> str:
    return (
        "The secret code for unit XJQ-482 is tulip-91. "
        "The secret code for unit XJQ-483 is tulip-19. "
        "The unit XJQ-482 was later selected for emergency access verification."
    )
