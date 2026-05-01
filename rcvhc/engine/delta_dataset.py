"""Small synthetic examples for Delta Memory experiments."""

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
    paired_sample_id: int | None = None
    collision_group_id: str | None = None
    foreign_answer: str | None = None
    address_text: str | None = None
    value_text: str | None = None
    foreign_address_text: str | None = None
    foreign_value_text: str | None = None
    address_char_range: list[int] | None = None
    value_char_range: list[int] | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


COLORS = ["tulip", "violet", "amber", "cedar", "indigo", "saffron", "lumen", "marble"]
SUFFIXES = ["91", "19", "17", "44", "63", "72", "38", "85"]
DELTA_TASK_SUITES = {
    "single_fact_late_reference",
    "multi_hop_binding",
    "temporal_overwrite",
    "paraphrase_nolima_style",
    "adversarial_negative",
    "paired_conflict_binding",
    "address_token_binding",
    "long_distance_nolima_style",
}


def make_delta_memory_examples(
    task_suite: str,
    num_examples: int,
    seed: int = 0,
    start_id: int = 0,
) -> list[DeltaExample]:
    if task_suite == "single_fact_late_reference":
        return make_later_reference_examples(num_examples, seed=seed, start_id=start_id)
    if task_suite == "multi_hop_binding":
        return make_multi_hop_binding_examples(num_examples, seed=seed, start_id=start_id)
    if task_suite == "temporal_overwrite":
        return make_temporal_overwrite_examples(num_examples, seed=seed, start_id=start_id)
    if task_suite == "paraphrase_nolima_style":
        return make_paraphrase_nolima_examples(num_examples, seed=seed, start_id=start_id)
    if task_suite == "adversarial_negative":
        return make_adversarial_negative_examples(num_examples, seed=seed, start_id=start_id)
    if task_suite == "paired_conflict_binding":
        return make_paired_conflict_binding_examples(num_examples, seed=seed, start_id=start_id)
    if task_suite == "address_token_binding":
        return make_address_token_binding_examples(num_examples, seed=seed, start_id=start_id)
    if task_suite == "long_distance_nolima_style":
        return make_long_distance_nolima_examples(num_examples, seed=seed, start_id=start_id)
    raise ValueError(f"unknown Delta Memory task suite: {task_suite}")


def make_later_reference_examples(num_examples: int, seed: int = 0, start_id: int = 0) -> list[DeltaExample]:
    rng = random.Random(seed)
    examples: list[DeltaExample] = []
    used_units: set[str] = set()
    used_codes: set[str] = set()
    for idx in range(num_examples):
        sample_id = start_id + idx
        unit = _unique_unit(rng, used_units)
        answer = _unique_code(rng, used_codes)
        distractors = []
        for d_idx in range(3):
            other = _unique_unit(rng, used_units)
            code = _unique_code(rng, used_codes)
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
        examples.append(DeltaExample(sample_id=sample_id, unit=unit, answer=answer, text=text, question=question, task_type="single_fact_late_reference"))
    return examples


def make_multi_hop_binding_examples(num_examples: int, seed: int = 0, start_id: int = 0) -> list[DeltaExample]:
    rng = random.Random(seed)
    examples: list[DeltaExample] = []
    used_units: set[str] = set()
    used_codes: set[str] = set()
    for idx in range(num_examples):
        sample_id = start_id + idx
        unit = _unique_unit(rng, used_units)
        badge = f"badge-{rng.randint(1000, 9999)}"
        locker = f"locker-{rng.randint(10, 99)}"
        answer = _unique_code(rng, used_codes)
        distractor_unit = _unique_unit(rng, used_units)
        text = "\n".join(
            [
                f"Unit {unit} was assigned access badge {badge}.",
                f"Badge {badge} maps to secure locker {locker}.",
                f"The recovery code inside {locker} is {answer}.",
                f"Unit {distractor_unit} was assigned access badge badge-{rng.randint(1000, 9999)}.",
                "Several routine access badges were audited without revealing recovery codes.",
                f"Later logs mention unit {unit} again but omit the badge, locker, and recovery code.",
            ]
        )
        question = f"What recovery code belongs to unit {unit}?"
        examples.append(DeltaExample(sample_id, unit, answer, text, question, "multi_hop_binding"))
    return examples


def make_temporal_overwrite_examples(num_examples: int, seed: int = 0, start_id: int = 0) -> list[DeltaExample]:
    rng = random.Random(seed)
    examples: list[DeltaExample] = []
    used_units: set[str] = set()
    used_codes: set[str] = set()
    for idx in range(num_examples):
        sample_id = start_id + idx
        unit = _unique_unit(rng, used_units)
        old_answer = _unique_code(rng, used_codes)
        answer = _unique_code(rng, used_codes)
        text = "\n".join(
            [
                f"Morning record: the secret code for unit {unit} is {old_answer}.",
                "A later maintenance event invalidated several morning records.",
                f"Evening record: the replacement secret code for unit {unit} is {answer}.",
                "The final query follows the latest-valid-record policy.",
                f"Unit {unit} appears in the final audit note without repeating the replacement code.",
            ]
        )
        question = f"Using the latest valid record, what is the secret code for unit {unit}?"
        examples.append(DeltaExample(sample_id, unit, answer, text, question, "temporal_overwrite"))
    return examples


def make_paraphrase_nolima_examples(num_examples: int, seed: int = 0, start_id: int = 0) -> list[DeltaExample]:
    rng = random.Random(seed)
    examples: list[DeltaExample] = []
    used_units: set[str] = set()
    used_codes: set[str] = set()
    for idx in range(num_examples):
        sample_id = start_id + idx
        unit = _unique_unit(rng, used_units)
        alias = f"north-aisle asset {rng.randint(100, 999)}"
        answer = _unique_code(rng, used_codes)
        text = "\n".join(
            [
                f"Inventory label {alias} corresponds to hardware unit {unit}.",
                f"The emergency phrase assigned to {alias} is {answer}.",
                "The phrase is not repeated in later notes, and the hardware ID is not restated near it.",
                f"A supervisor later asked about hardware unit {unit} after reading unrelated status notes.",
            ]
        )
        question = f"What emergency phrase is associated with hardware unit {unit}?"
        examples.append(DeltaExample(sample_id, unit, answer, text, question, "paraphrase_nolima_style"))
    return examples


def make_adversarial_negative_examples(num_examples: int, seed: int = 0, start_id: int = 0) -> list[DeltaExample]:
    rng = random.Random(seed)
    examples: list[DeltaExample] = []
    used_units: set[str] = set()
    used_codes: set[str] = set()
    for idx in range(num_examples):
        sample_id = start_id + idx
        unit = _unique_unit(rng, used_units)
        near_unit = f"{unit[:-1]}{(int(unit[-1]) + 1) % 10}"
        used_units.add(near_unit)
        answer = _unique_code(rng, used_codes)
        wrong = _unique_code(rng, used_codes)
        text = "\n".join(
            [
                f"The secret code for unit {near_unit} is {wrong}.",
                f"The secret code for unit {unit} is {answer}.",
                f"Both {unit} and {near_unit} appear in later verification notes with similar formatting.",
                "The final note deliberately places the near-match unit beside the true unit.",
                f"Verifier target: unit {unit}.",
            ]
        )
        question = f"What is the secret code for unit {unit}?"
        examples.append(DeltaExample(sample_id, unit, answer, text, question, "adversarial_negative"))
    return examples


def make_paired_conflict_binding_examples(num_examples: int, seed: int = 0, start_id: int = 0) -> list[DeltaExample]:
    rng = random.Random(seed)
    examples: list[DeltaExample] = []
    used_units: set[str] = set()
    used_codes: set[str] = set()
    used_cases: set[str] = set()
    while len(examples) < num_examples:
        unit = _unique_unit(rng, used_units)
        pair = []
        for _ in range(2):
            case_id = _unique_case(rng, used_cases)
            answer = _unique_code(rng, used_codes)
            pair.append((case_id, answer))
        sample_ids = [start_id + len(examples) + offset for offset in range(2)]
        group_id = f"paired-conflict-{unit}"
        for pair_idx, (case_id, answer) in enumerate(pair):
            if len(examples) >= num_examples:
                break
            sample_id = sample_ids[pair_idx]
            paired_sample_id = sample_ids[1 - pair_idx] if len(examples) + 1 < num_examples or pair_idx == 1 else None
            foreign_answer = pair[1 - pair_idx][1] if paired_sample_id is not None else None
            distractor_unit = _unique_unit(rng, used_units)
            distractor_case = _unique_case(rng, used_cases)
            distractor_answer = _unique_code(rng, used_codes)
            text = "\n".join(
                [
                    f"Ledger {case_id}: unit {unit} was assigned secret code {answer}.",
                    f"Ledger {distractor_case}: unit {distractor_unit} was assigned secret code {distractor_answer}.",
                    f"The audit later references ledger {case_id} and unit {unit} without repeating the code.",
                    "The final lookup must use both the ledger identifier and the unit identifier.",
                ]
            )
            question = f"For ledger {case_id}, what is the secret code for unit {unit}?"
            examples.append(
                DeltaExample(
                    sample_id,
                    unit,
                    answer,
                    text,
                    question,
                    "paired_conflict_binding",
                    paired_sample_id=paired_sample_id,
                    collision_group_id=group_id,
                    foreign_answer=foreign_answer,
                )
            )
    return examples


def make_address_token_binding_examples(num_examples: int, seed: int = 0, start_id: int = 0) -> list[DeltaExample]:
    rng = random.Random(seed)
    examples: list[DeltaExample] = []
    used_units: set[str] = set()
    used_codes: set[str] = set()
    used_cases: set[str] = set()
    while len(examples) < num_examples:
        unit = _unique_unit(rng, used_units)
        pair = []
        for _ in range(2):
            case_id = _unique_case(rng, used_cases)
            answer = _unique_code(rng, used_codes)
            address = f"ADDR::{case_id}::{unit}"
            pair.append((case_id, address, answer))
        sample_ids = [start_id + len(examples) + offset for offset in range(2)]
        group_id = f"address-token-{unit}"
        for pair_idx, (case_id, address, answer) in enumerate(pair):
            if len(examples) >= num_examples:
                break
            paired_sample_id = sample_ids[1 - pair_idx] if len(examples) + 1 < num_examples or pair_idx == 1 else None
            foreign_answer = pair[1 - pair_idx][2] if paired_sample_id is not None else None
            foreign_address = pair[1 - pair_idx][1] if paired_sample_id is not None else None
            foreign_value = f"secret-code = {foreign_answer}" if foreign_answer is not None else None
            value_text = f"secret-code = {answer}"
            text = "\n".join(
                [
                    "Memory card format: each card has an ADDRESS line and one PAYLOAD line.",
                    f"ADDRESS: {address}",
                    f"PAYLOAD: {value_text}",
                    f"VALIDATION: use the full address {address}; do not answer from unit id alone.",
                    "The payload is intentionally not repeated outside this address card.",
                ]
            )
            address_start = text.index(address)
            value_start = text.index(value_text)
            question = f"For ADDRESS {address}, what is the secret-code payload?"
            examples.append(
                DeltaExample(
                    sample_ids[pair_idx],
                    unit,
                    answer,
                    text,
                    question,
                    "address_token_binding",
                    paired_sample_id=paired_sample_id,
                    collision_group_id=group_id,
                    foreign_answer=foreign_answer,
                    address_text=address,
                    value_text=value_text,
                    foreign_address_text=foreign_address,
                    foreign_value_text=foreign_value,
                    address_char_range=[address_start, address_start + len(address)],
                    value_char_range=[value_start, value_start + len(value_text)],
                )
            )
    return examples


def make_long_distance_nolima_examples(num_examples: int, seed: int = 0, start_id: int = 0) -> list[DeltaExample]:
    rng = random.Random(seed)
    examples: list[DeltaExample] = []
    used_units: set[str] = set()
    used_codes: set[str] = set()
    for idx in range(num_examples):
        sample_id = start_id + idx
        unit = _unique_unit(rng, used_units)
        alias = f"regional asset {rng.randint(1000, 9999)}"
        answer = _unique_code(rng, used_codes)
        filler = []
        for filler_idx in range(18):
            distractor = _unique_unit(rng, used_units)
            filler.append(
                " ".join(
                    [
                        f"Archive paragraph {filler_idx}: unit {distractor} passed routine calibration.",
                        "The paragraph discusses inventories, weather delays, route planning, and maintenance logs.",
                        "No emergency phrase, code phrase, ledger answer, or recovery token is provided here.",
                    ]
                )
            )
        text = "\n".join(
            [
                f"Crosswalk note: hardware identifier {unit} is referred to as {alias} in the regional archive.",
                *filler[:9],
                f"Policy memorandum: the emergency phrase assigned to {alias} is {answer}.",
                "The phrase is intentionally not repeated after this memorandum.",
                *filler[9:],
                f"Final audit note: hardware identifier {unit} requires the archived emergency phrase.",
            ]
        )
        question = f"What emergency phrase is assigned to hardware identifier {unit}?"
        examples.append(DeltaExample(sample_id, unit, answer, text, question, "long_distance_nolima_style"))
    return examples


def _unique_unit(rng: random.Random, used: set[str]) -> str:
    while True:
        unit = f"XJQ-{rng.randint(100, 999)}"
        if unit not in used:
            used.add(unit)
            return unit


def _unique_case(rng: random.Random, used: set[str]) -> str:
    while True:
        case_id = f"case-{rng.randint(1000, 9999)}"
        if case_id not in used:
            used.add(case_id)
            return case_id


def _unique_code(rng: random.Random, used: set[str]) -> str:
    while True:
        code = f"{rng.choice(COLORS)}-{rng.randint(10, 99)}"
        if code not in used:
            used.add(code)
            return code
