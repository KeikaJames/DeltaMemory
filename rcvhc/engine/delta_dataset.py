"""Small synthetic examples for Delta attention-memory experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class DeltaExample:
    sample_id: int
    unit: str
    answer: str
    text: str
    question: str
    task_type: str = "later_referenced_fact"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


COLORS = ["tulip", "violet", "amber", "cedar", "indigo", "saffron", "lumen", "marble"]
SUFFIXES = ["91", "19", "17", "44", "63", "72", "38", "85"]


def make_later_reference_examples(num_examples: int, seed: int = 0, start_id: int = 0) -> list[DeltaExample]:
    rng = random.Random(seed)
    examples: list[DeltaExample] = []
    used_units: set[str] = set()
    for idx in range(num_examples):
        sample_id = start_id + idx
        unit = _unique_unit(rng, used_units)
        answer = f"{COLORS[idx % len(COLORS)]}-{SUFFIXES[(idx * 3) % len(SUFFIXES)]}"
        distractors = []
        for d_idx in range(3):
            other = _unique_unit(rng, used_units)
            code = f"{COLORS[(idx + d_idx + 1) % len(COLORS)]}-{SUFFIXES[(idx + d_idx + 2) % len(SUFFIXES)]}"
            distractors.append(f"The secret code for unit {other} is {code}.")
        filler = " ".join(
            [
                "Operations continued with routine inventory checks.",
                "The staff logged unrelated pressure readings and shift notes.",
                "No code value was repeated in this operational section.",
            ]
        )
        text = "\n".join(
            [
                f"The secret code for unit {unit} is {answer}.",
                *distractors,
                filler,
                f"The unit {unit} was later selected for emergency access verification.",
                f"The verifier repeatedly checked unit {unit} before the final query.",
                f"The final operator note names unit {unit} again but deliberately does not repeat the code value.",
            ]
        )
        question = f"What is the secret code for unit {unit}?"
        examples.append(DeltaExample(sample_id=sample_id, unit=unit, answer=answer, text=text, question=question))
    return examples


def _unique_unit(rng: random.Random, used: set[str]) -> str:
    while True:
        unit = f"XJQ-{rng.randint(100, 999)}"
        if unit not in used:
            used.add(unit)
            return unit
