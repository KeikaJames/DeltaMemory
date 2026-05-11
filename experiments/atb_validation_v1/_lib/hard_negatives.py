"""Hard-negative bank construction for Exp13 / Exp14 / Exp20.

Builds maps from CounterFact rows to lists of "hard" alternative fact_ids:

  * same_subject_wrong_object  : facts sharing the subject string
  * same_relation_wrong_subject: facts sharing the relation id (P-code)
  * same_object_wrong_subject  : facts sharing target_new

Used by addressability and oracle-readout phases to test whether the
correct fact's M_K is preferentially scored over plausible distractors,
not just random ones.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class HardNegativeIndex:
    """Lookup tables built once from a CounterFact subset."""
    by_subject: dict[str, list[str]] = field(default_factory=dict)
    by_relation: dict[str, list[str]] = field(default_factory=dict)
    by_target_new: dict[str, list[str]] = field(default_factory=dict)

    def neighbors(self, row: dict, kind: str) -> list[str]:
        fid = row.get("id")
        if kind == "same_subject_wrong_object":
            pool = self.by_subject.get(row.get("subject", ""), [])
        elif kind == "same_relation_wrong_subject":
            pool = self.by_relation.get(row.get("relation", ""), [])
        elif kind == "same_object_wrong_subject":
            pool = self.by_target_new.get(row.get("target_new", ""), [])
        else:
            raise ValueError(f"unknown hard-negative kind: {kind!r}")
        return [f for f in pool if f != fid]


def build_hard_negatives(rows: list[dict]) -> HardNegativeIndex:
    """Index a list of CounterFact rows for cheap neighbor queries."""
    idx = HardNegativeIndex()
    by_subject: dict[str, list[str]] = defaultdict(list)
    by_relation: dict[str, list[str]] = defaultdict(list)
    by_target_new: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        fid = r.get("id")
        if fid is None:
            continue
        by_subject[r.get("subject", "")].append(fid)
        by_relation[r.get("relation", "")].append(fid)
        by_target_new[r.get("target_new", "")].append(fid)
    idx.by_subject = dict(by_subject)
    idx.by_relation = dict(by_relation)
    idx.by_target_new = dict(by_target_new)
    return idx
