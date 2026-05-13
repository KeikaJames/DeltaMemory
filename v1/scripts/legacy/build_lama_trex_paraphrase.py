#!/usr/bin/env python3
"""Build paraphrase variants of LAMA-TREx prompts for Stage 10A.

Output JSONL schema: each line is one fact with 6 paraphrase templates of
the read prompt. The first paraphrase is the canonical prompt (matches
what's in lama_trex_full.jsonl); the remaining five are surface-form
variations preserving the same gold answer token.

Usage:
    python3 scripts/build_lama_trex_paraphrase.py
"""
from __future__ import annotations

import json
from pathlib import Path

# Paraphrase templates per Wikidata relation.
# Each template uses {entity} placeholder. Canonical = templates[0].
RELATION_TEMPLATES: dict[str, list[str]] = {
    "P36": [  # capital
        "The capital of {entity} is",
        "{entity}'s capital city is",
        "The capital city of {entity}, named",
        "Q: What is the capital of {entity}? A:",
        "{entity} has its capital at",
        "Located in {entity}, the capital is",
    ],
    "P19": [  # place of birth
        "{entity} was born in",
        "The birthplace of {entity} is",
        "{entity} was born in the city of",
        "Q: Where was {entity} born? A:",
        "Born in",
        "{entity}'s place of birth is",
    ],
    "P101": [  # field of work
        "{entity} works in the field of",
        "The field of work of {entity} is",
        "{entity}'s area of expertise is",
        "Q: What field does {entity} work in? A:",
        "{entity} specializes in",
        "The discipline of {entity} is",
    ],
    "P641": [  # sport
        "{entity} plays the sport of",
        "The sport of {entity} is",
        "{entity} is a player of",
        "Q: What sport does {entity} play? A:",
        "{entity} competes in",
        "The athletic discipline of {entity} is",
    ],
    "P140": [  # religion
        "The religion of {entity} is",
        "{entity} follows the religion of",
        "{entity} practices",
        "Q: What religion does {entity} follow? A:",
        "{entity}'s religious affiliation is",
        "The faith of {entity} is",
    ],
    "P39": [  # position held
        "{entity} holds the position of",
        "The position held by {entity} is",
        "{entity} serves as",
        "Q: What position does {entity} hold? A:",
        "{entity}'s role is",
        "The office held by {entity} is",
    ],
    "P937": [  # work location
        "{entity} works in",
        "The work location of {entity} is",
        "{entity} is based in",
        "Q: Where does {entity} work? A:",
        "{entity}'s work city is",
        "Located in",
    ],
}


def _detect_relation(prompt: str, value: str) -> str | None:
    """Best-effort backsolve of a fact's relation by canonical prompt fragment."""
    pl = prompt.lower()
    if "capital of" in pl or "capital city" in pl:
        return "P36"
    if "born in" in pl or "birthplace" in pl:
        return "P19"
    if "field of" in pl or "area of expertise" in pl or "discipline" in pl or "taught" in pl:
        return "P101"
    if "sport of" in pl or " sport" in pl or "competes in" in pl or "plays the sport" in pl:
        return "P641"
    if "religion" in pl or "faith" in pl or "practises" in pl or "practices" in pl:
        return "P140"
    if "position of" in pl or "office of" in pl or "serves as" in pl or "worked as" in pl or "works as" in pl or "holds the position" in pl:
        return "P39"
    if "works in" in pl or "based in" in pl or "work location" in pl or "worked in" in pl:
        return "P937"
    return None


def _detect_entity(prompt: str) -> str | None:
    """Heuristic entity extraction from canonical prompts.

    Strips known suffixes; if no suffix matches, returns the run of words
    before the first detected predicate fragment.
    """
    p = prompt.strip().rstrip(":").rstrip()
    suffixes = [
        " was born in", " worked in the field of", " worked in",
        " worked as a", " worked as an", " works as a", " works as an",
        " works in the field of", " works in",
        " plays the sport of", " played the sport of",
        " competes in the sport of", " competes in",
        " practises the religion of", " practices the religion of",
        " holds the position of", " held the position of", " serves as",
        " taught",
    ]
    for s in suffixes:
        if p.endswith(s):
            return p[:-len(s)].strip()
    # capital templates
    if p.startswith("The capital of "):
        rest = p[len("The capital of "):]
        if rest.endswith(" is"):
            rest = rest[:-3]
        return rest.strip()
    if p.startswith("The field of work of "):
        rest = p[len("The field of work of "):]
        if rest.endswith(" is"):
            rest = rest[:-3]
        return rest.strip()
    if p.startswith("The religion of "):
        rest = p[len("The religion of "):]
        if rest.endswith(" is"):
            rest = rest[:-3]
        return rest.strip()
    return None


def main() -> int:
    src = Path("scripts/data/lama_trex_full.jsonl")
    dst = Path("scripts/data/lama_trex_paraphrase.jsonl")
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    out = []
    skipped = 0
    for row in rows:
        addr = row["address"]
        val = row["value"]
        rel = _detect_relation(addr, val)
        ent = _detect_entity(addr)
        if rel is None or ent is None:
            skipped += 1
            continue
        templates = RELATION_TEMPLATES[rel]
        paraphrases = [t.format(entity=ent) for t in templates]
        out.append({
            "address_canonical": paraphrases[0],
            "address": addr,  # keep original key for compat
            "value": val,
            "relation": rel,
            "entity": ent,
            "paraphrases": paraphrases,
        })
    with dst.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"[paraphrase] wrote {len(out)} facts ({skipped} skipped) to {dst}")
    by_rel: dict[str, int] = {}
    for r in out:
        by_rel[r["relation"]] = by_rel.get(r["relation"], 0) + 1
    print(f"[paraphrase] by relation: {by_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
