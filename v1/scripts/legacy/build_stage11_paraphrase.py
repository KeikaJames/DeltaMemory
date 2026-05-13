#!/usr/bin/env python3
"""Stage 11 paraphrase pool: 10 templates per relation, train=6 / holdout=4.

Templates 0-5 are reused from Stage 10 (familiar surface forms). Templates
6-9 are *structurally novel* (passive voice, negation-style hedging,
question-with-context, multi-clause), so the held-out set is genuinely OOD
in surface form, not just in lexical choice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_lama_trex_paraphrase import RELATION_TEMPLATES as STAGE10_TEMPLATES
from build_lama_trex_paraphrase import _detect_entity, _detect_relation

# Structurally novel held-out templates (4 per relation). These use
# passive voice, multi-clause, or rare framings to force surface-OOD.
HOLDOUT_TEMPLATES: dict[str, list[str]] = {
    "P36": [
        "When asked about {entity}, one usually names its capital,",
        "Among the world's capitals, the one belonging to {entity} is named",
        "The seat of government of the country known as {entity} is the city of",
        "If you were to visit {entity}, the principal city you would land in is",
    ],
    "P19": [
        "Records indicate that {entity} first drew breath in",
        "The municipality where {entity} entered the world is",
        "If asked about the origins of {entity}, one would point to",
        "Biographers note that {entity} hails from",
    ],
    "P101": [
        "Scholarly contributions of {entity} fall under the discipline of",
        "When categorising research, {entity} is associated with",
        "The intellectual domain in which {entity} operates is",
        "Textbooks classify {entity} under the field known as",
    ],
    "P641": [
        "Athletic records show {entity} competing in",
        "The competitive arena in which {entity} performs is",
        "Statisticians group {entity} with athletes of",
        "If you switch on a broadcast featuring {entity}, you'll be watching",
    ],
    "P140": [
        "Religious surveys list {entity} as a follower of",
        "The spiritual tradition adhered to by {entity} is",
        "Among world faiths, the one practised by {entity} is",
        "When discussing the beliefs of {entity}, one refers to",
    ],
    "P39": [
        "Official records register {entity} as occupant of the office of",
        "The institutional role attached to {entity} carries the title of",
        "When listing officeholders, {entity} appears under",
        "Civic registers describe {entity} as serving as",
    ],
    "P937": [
        "Professional rosters place {entity} working out of",
        "The metropolis from which {entity} conducts work is",
        "Workplace records associate {entity} with the city of",
        "If you were to visit {entity}'s office, you would travel to",
    ],
}


def main() -> int:
    src = Path("scripts/data/lama_trex_full.jsonl")
    dst_train = Path("scripts/data/lama_stage11_train_paraphrase.jsonl")
    dst_holdout = Path("scripts/data/lama_stage11_holdout_paraphrase.jsonl")
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    train_out, holdout_out = [], []
    skipped = 0
    for row in rows:
        addr = row["address"]
        val = row["value"]
        rel = _detect_relation(addr, val)
        ent = _detect_entity(addr)
        if rel is None or ent is None or rel not in STAGE10_TEMPLATES or rel not in HOLDOUT_TEMPLATES:
            skipped += 1
            continue
        train_paras = [t.format(entity=ent) for t in STAGE10_TEMPLATES[rel]]
        holdout_paras = [t.format(entity=ent) for t in HOLDOUT_TEMPLATES[rel]]
        train_out.append({
            "address_canonical": train_paras[0], "address": addr, "value": val,
            "relation": rel, "entity": ent, "paraphrases": train_paras,
        })
        holdout_out.append({
            "address_canonical": train_paras[0], "address": addr, "value": val,
            "relation": rel, "entity": ent, "paraphrases": holdout_paras,
        })
    with dst_train.open("w") as f:
        for r in train_out:
            f.write(json.dumps(r) + "\n")
    with dst_holdout.open("w") as f:
        for r in holdout_out:
            f.write(json.dumps(r) + "\n")
    by_rel = {}
    for r in train_out:
        by_rel[r["relation"]] = by_rel.get(r["relation"], 0) + 1
    print(f"[stage11] train={len(train_out)} holdout={len(holdout_out)} skipped={skipped}")
    print(f"[stage11] by relation: {by_rel}")
    print(f"[stage11] train templates / fact = 6 (Stage 10 reused)")
    print(f"[stage11] holdout templates / fact = 4 (structurally novel, surface-OOD)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
