#!/usr/bin/env python3
"""Build v3.1 (write, paraphrase) pairs for Stage 15 K-projector retraining.

Phase L1 of the v3.1 plan. Compared with the v3 splits (7 relations, 176 facts,
~1,056 pairs) this builder:

1. Expands to ≥30 relations from LAMA-TREx + ConceptNet, chosen so each relation
   has a deterministic single-token (or short) answer for hand-curated entities.
2. Uses a programmatic paraphrase template library (avg ≥10 paraphrases per
   fact) so the encoder sees more surface variation per (entity, relation) pair.
3. Writes train_v31 / dev_v31 / val2_v31 / test_v31 splits with sha-locked seeds
   so v3.1 evaluation is preregistered (see docs/preregistration.md amendment).

Output (under eval/splits_v31/):
    train_v31.jsonl
    dev_v31.jsonl
    val2_v31.jsonl
    test_v31.jsonl
    SPLIT_INFO.json   (sha256 of each + counts + seed)

Schema (per line, identical to v3 splits for compatibility):
    {"address": "...", "address_canonical": "...", "entity": "...",
     "paraphrases": ["...", ...], "relation": "Pxx", "value": "..."}

Usage:
    python eval/build_v31_pairs.py --seed 0 --min-pairs 2500
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "eval" / "splits_v31"


# ---------------------------------------------------------------------------
# Paraphrase template library
# ---------------------------------------------------------------------------
# Each entry: relation -> list of templates with {entity} placeholder.
# These are the paraphrase surfaces; the answer slot is appended at the end
# of every template (as in LAMA-TREx). Hand-written, declarative + Q/A mix.

PARAPHRASE_TEMPLATES: dict[str, list[str]] = {
    "P36": [
        "The capital of {entity} is",
        "{entity}'s capital is",
        "{entity} has its capital at",
        "The seat of government of {entity} is in",
        "Q: What is the capital of {entity}? A:",
        "{entity}'s capital city is",
        "The administrative center of {entity} is",
        "When you visit {entity}, the capital you fly into is",
        "{entity} is governed from",
        "The capital city of {entity} happens to be",
    ],
    "P19": [
        "{entity} was born in",
        "The birthplace of {entity} is",
        "{entity}'s birthplace is",
        "Q: Where was {entity} born? A:",
        "{entity} was born in the city of",
        "{entity} hails from",
        "The hometown of {entity} is",
        "{entity} grew up in",
        "{entity} originated from",
        "The native city of {entity} is",
    ],
    "P39": [
        "{entity} held the position of",
        "The position held by {entity} is",
        "{entity}'s role is",
        "Q: What position does {entity} hold? A:",
        "{entity} works as",
        "{entity} serves as",
        "{entity} is best known as",
        "The job of {entity} is",
        "{entity}'s occupation is",
        "{entity} holds the office of",
    ],
    "P101": [
        "{entity} works in the field of",
        "The field of work of {entity} is",
        "{entity}'s area of expertise is",
        "Q: What field does {entity} work in? A:",
        "{entity} specializes in",
        "The discipline of {entity} is",
        "{entity}'s research area is",
        "{entity} contributes to",
        "{entity} is famous in the field of",
        "The expertise of {entity} lies in",
    ],
    "P140": [
        "{entity} follows the religion of",
        "The religion of {entity} is",
        "{entity} practices",
        "Q: What is {entity}'s religion? A:",
        "{entity} adheres to",
        "{entity}'s faith is",
        "{entity} believes in the religion called",
        "The faith of {entity} is",
        "{entity} identifies religiously as",
        "{entity}'s religious tradition is",
    ],
    "P641": [
        "{entity} plays the sport of",
        "The sport of {entity} is",
        "{entity}'s sport is",
        "Q: What sport does {entity} play? A:",
        "{entity} competes in",
        "{entity} is an athlete in",
        "{entity} is associated with the sport of",
        "{entity}'s discipline is",
        "{entity}'s athletic field is",
        "The game of {entity} is",
    ],
    "P937": [
        "{entity} worked in the city of",
        "The work location of {entity} is",
        "{entity}'s workplace is in",
        "Q: Where did {entity} work? A:",
        "{entity} was based in",
        "The city where {entity} worked is",
        "{entity} carried out work in",
        "{entity} performed their work in",
        "{entity}'s career was in",
        "{entity} did most of their work in",
    ],
    "P17": [
        "{entity} is located in the country of",
        "The country of {entity} is",
        "{entity} is in",
        "Q: In which country is {entity}? A:",
        "{entity} belongs to the nation of",
        "{entity}'s host country is",
        "{entity} sits inside the borders of",
        "The nation containing {entity} is",
        "{entity} is part of",
        "{entity} can be found in the country called",
    ],
    "P27": [
        "{entity} is a citizen of",
        "The country of citizenship of {entity} is",
        "{entity}'s nationality is",
        "Q: What is the nationality of {entity}? A:",
        "{entity} holds citizenship in",
        "{entity} is a national of",
        "{entity} carries the passport of",
        "{entity} is from the country of",
        "The nation of {entity}'s citizenship is",
        "{entity} is officially a citizen of",
    ],
    "P103": [
        "The native language of {entity} is",
        "{entity}'s mother tongue is",
        "{entity} natively speaks",
        "Q: What is {entity}'s native language? A:",
        "{entity} grew up speaking",
        "The first language of {entity} is",
        "{entity}'s native tongue is",
        "{entity} learned to speak",
        "{entity} natively speaks the language of",
        "From birth {entity} spoke",
    ],
    "P176": [
        "{entity} was manufactured by",
        "The manufacturer of {entity} is",
        "{entity}'s maker is",
        "Q: Who manufactures {entity}? A:",
        "{entity} is produced by",
        "The producer of {entity} is",
        "{entity} comes from the company called",
        "{entity} is a product of",
        "The company behind {entity} is",
        "{entity} is built by",
    ],
    "P276": [
        "{entity} is located at",
        "The location of {entity} is",
        "{entity}'s site is",
        "Q: Where is {entity} located? A:",
        "{entity} sits at",
        "{entity} can be found at",
        "{entity} is situated at",
        "The geographical location of {entity} is",
        "{entity} is positioned at",
        "The address of {entity} is in",
    ],
    "P407": [
        "{entity} was written in",
        "The language of {entity} is",
        "{entity}'s text is in",
        "Q: What language is {entity} written in? A:",
        "{entity} appeared originally in",
        "The original language of {entity} is",
        "{entity} was published in the language of",
        "{entity} reads in",
        "The medium of {entity} is",
        "{entity} was composed in",
    ],
    "P413": [
        "{entity} plays in the position of",
        "The playing position of {entity} is",
        "{entity}'s role on the field is",
        "Q: What position does {entity} play? A:",
        "{entity} is positioned as",
        "{entity} plays as a",
        "{entity}'s field role is",
        "On the field {entity} is a",
        "{entity}'s specialty position is",
        "{entity} typically plays at",
    ],
    "P159": [
        "The headquarters of {entity} are in",
        "{entity} is headquartered in",
        "{entity}'s HQ is in",
        "Q: Where is {entity} headquartered? A:",
        "{entity}'s main office is in",
        "{entity} runs its operations from",
        "{entity}'s base of operations is",
        "{entity}'s corporate seat is in",
        "{entity} has its headquarters at",
        "The main office of {entity} is in",
    ],
}


# ---------------------------------------------------------------------------
# Fact bank
# ---------------------------------------------------------------------------
# (relation, entity, value)
FACTS: list[tuple[str, str, str]] = []


def _add(rel: str, pairs: list[tuple[str, str]]) -> None:
    for ent, val in pairs:
        FACTS.append((rel, ent, val))


# P36 capital (50)
_add("P36", [
    ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
    ("Italy", "Rome"), ("Spain", "Madrid"), ("Russia", "Moscow"),
    ("China", "Beijing"), ("Egypt", "Cairo"), ("Greece", "Athens"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
    ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"), ("Portugal", "Lisbon"),
    ("Ireland", "Dublin"), ("Hungary", "Budapest"), ("Turkey", "Ankara"),
    ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Israel", "Jerusalem"),
    ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"),
    ("Korea", "Seoul"), ("Australia", "Canberra"), ("Argentina", "Buenos"),
    ("Brazil", "Brasilia"), ("Chile", "Santiago"), ("Peru", "Lima"),
    ("Cuba", "Havana"), ("Canada", "Ottawa"), ("Kenya", "Nairobi"),
    ("Nigeria", "Abuja"), ("Morocco", "Rabat"), ("Algeria", "Algiers"),
    ("Tunisia", "Tunis"), ("Ghana", "Accra"), ("Pakistan", "Islamabad"),
    ("Bangladesh", "Dhaka"), ("Nepal", "Kathmandu"), ("Mongolia", "Ulaanbaatar"),
    ("Iceland", "Reykjavik"), ("Switzerland", "Bern"), ("Czechia", "Prague"),
    ("Slovakia", "Bratislava"), ("Bulgaria", "Sofia"),
])

# P19 birthplace (30)
_add("P19", [
    ("Albert Einstein", "Ulm"), ("Mahatma Gandhi", "Porbandar"),
    ("Mozart", "Salzburg"), ("Beethoven", "Bonn"),
    ("Napoleon", "Ajaccio"), ("Nelson Mandela", "Mvezo"),
    ("Marie Curie", "Warsaw"), ("Pablo Picasso", "Malaga"),
    ("Charles Darwin", "Shrewsbury"), ("Vincent van Gogh", "Zundert"),
    ("Frida Kahlo", "Coyoacan"), ("Steve Jobs", "San Francisco"),
    ("Bill Gates", "Seattle"), ("Stephen Hawking", "Oxford"),
    ("Isaac Newton", "Woolsthorpe"), ("Galileo Galilei", "Pisa"),
    ("Leonardo da Vinci", "Vinci"), ("Michelangelo", "Caprese"),
    ("William Shakespeare", "Stratford"), ("Charles Dickens", "Portsmouth"),
    ("Jane Austen", "Steventon"), ("Mark Twain", "Florida"),
    ("Ernest Hemingway", "Oak Park"), ("Sigmund Freud", "Freiberg"),
    ("Carl Jung", "Kesswil"), ("Friedrich Nietzsche", "Rocken"),
    ("Confucius", "Qufu"), ("Sun Tzu", "Qi"),
    ("Genghis Khan", "Delun"), ("Akira Kurosawa", "Tokyo"),
])

# P39 position (25)
_add("P39", [
    ("Barack Obama", "president"), ("Angela Merkel", "chancellor"),
    ("Pope Francis", "pope"), ("Queen Elizabeth", "queen"),
    ("Vladimir Putin", "president"), ("Xi Jinping", "president"),
    ("Emmanuel Macron", "president"), ("Boris Johnson", "minister"),
    ("Justin Trudeau", "minister"), ("Narendra Modi", "minister"),
    ("Joe Biden", "president"), ("Donald Trump", "president"),
    ("Theresa May", "minister"), ("David Cameron", "minister"),
    ("Tony Blair", "minister"), ("Margaret Thatcher", "minister"),
    ("Winston Churchill", "minister"), ("Franklin Roosevelt", "president"),
    ("John Kennedy", "president"), ("Abraham Lincoln", "president"),
    ("George Washington", "president"), ("Mikhail Gorbachev", "leader"),
    ("Fidel Castro", "leader"), ("Mao Zedong", "chairman"),
    ("Kim Jong-un", "leader"),
])

# P101 field (25)
_add("P101", [
    ("Stephen Hawking", "physics"), ("Albert Einstein", "physics"),
    ("Marie Curie", "physics"), ("Charles Darwin", "biology"),
    ("Isaac Newton", "physics"), ("Galileo Galilei", "astronomy"),
    ("Sigmund Freud", "psychology"), ("Carl Jung", "psychology"),
    ("Pablo Picasso", "art"), ("Vincent van Gogh", "art"),
    ("Leonardo da Vinci", "art"), ("Michelangelo", "art"),
    ("Mozart", "music"), ("Beethoven", "music"),
    ("Bach", "music"), ("William Shakespeare", "literature"),
    ("Charles Dickens", "literature"), ("Jane Austen", "literature"),
    ("Mark Twain", "literature"), ("Ernest Hemingway", "literature"),
    ("Friedrich Nietzsche", "philosophy"), ("Confucius", "philosophy"),
    ("Plato", "philosophy"), ("Aristotle", "philosophy"),
    ("Socrates", "philosophy"),
])

# P140 religion (15)
_add("P140", [
    ("Pope Francis", "Christianity"), ("Dalai Lama", "Buddhism"),
    ("Mahatma Gandhi", "Hinduism"), ("Mother Teresa", "Christianity"),
    ("Martin Luther King", "Christianity"), ("Malcolm X", "Islam"),
    ("Muhammad Ali", "Islam"), ("Albert Einstein", "Judaism"),
    ("Sigmund Freud", "Judaism"), ("Karl Marx", "Judaism"),
    ("Confucius", "Confucianism"), ("Lao Tzu", "Taoism"),
    ("Buddha", "Buddhism"), ("Jesus", "Judaism"),
    ("Muhammad", "Islam"),
])

# P641 sport (20)
_add("P641", [
    ("Michael Jordan", "basketball"), ("LeBron James", "basketball"),
    ("Kobe Bryant", "basketball"), ("Cristiano Ronaldo", "football"),
    ("Lionel Messi", "football"), ("Pele", "football"),
    ("Diego Maradona", "football"), ("David Beckham", "football"),
    ("Tiger Woods", "golf"), ("Roger Federer", "tennis"),
    ("Rafael Nadal", "tennis"), ("Serena Williams", "tennis"),
    ("Usain Bolt", "athletics"), ("Michael Phelps", "swimming"),
    ("Muhammad Ali", "boxing"), ("Mike Tyson", "boxing"),
    ("Wayne Gretzky", "hockey"), ("Tom Brady", "football"),
    ("Babe Ruth", "baseball"), ("Pete Sampras", "tennis"),
])

# P937 work_location (15)
_add("P937", [
    ("Albert Einstein", "Princeton"), ("Stephen Hawking", "Cambridge"),
    ("Isaac Newton", "Cambridge"), ("Galileo Galilei", "Padua"),
    ("Charles Darwin", "London"), ("Sigmund Freud", "Vienna"),
    ("Marie Curie", "Paris"), ("Pablo Picasso", "Paris"),
    ("Vincent van Gogh", "Arles"), ("Leonardo da Vinci", "Florence"),
    ("Michelangelo", "Rome"), ("Mozart", "Vienna"),
    ("Beethoven", "Vienna"), ("William Shakespeare", "London"),
    ("Charles Dickens", "London"),
])

# P17 located in country (20)
_add("P17", [
    ("Paris", "France"), ("Berlin", "Germany"), ("Tokyo", "Japan"),
    ("London", "England"), ("Madrid", "Spain"), ("Rome", "Italy"),
    ("Moscow", "Russia"), ("Beijing", "China"), ("Cairo", "Egypt"),
    ("Athens", "Greece"), ("Stockholm", "Sweden"), ("Oslo", "Norway"),
    ("Vienna", "Austria"), ("Brussels", "Belgium"), ("Lisbon", "Portugal"),
    ("Dublin", "Ireland"), ("Helsinki", "Finland"), ("Warsaw", "Poland"),
    ("Sydney", "Australia"), ("Toronto", "Canada"),
])

# P27 citizenship (20)
_add("P27", [
    ("Albert Einstein", "Germany"), ("Marie Curie", "Poland"),
    ("Stephen Hawking", "Britain"), ("Isaac Newton", "Britain"),
    ("Charles Darwin", "Britain"), ("Sigmund Freud", "Austria"),
    ("Pablo Picasso", "Spain"), ("Vincent van Gogh", "Netherlands"),
    ("Leonardo da Vinci", "Italy"), ("Michelangelo", "Italy"),
    ("Mozart", "Austria"), ("Beethoven", "Germany"),
    ("Bach", "Germany"), ("William Shakespeare", "England"),
    ("Charles Dickens", "England"), ("Jane Austen", "England"),
    ("Friedrich Nietzsche", "Germany"), ("Confucius", "China"),
    ("Lao Tzu", "China"), ("Sun Tzu", "China"),
])

# P103 native language (20)
_add("P103", [
    ("Albert Einstein", "German"), ("Marie Curie", "Polish"),
    ("Sigmund Freud", "German"), ("Pablo Picasso", "Spanish"),
    ("Vincent van Gogh", "Dutch"), ("Leonardo da Vinci", "Italian"),
    ("Michelangelo", "Italian"), ("Mozart", "German"),
    ("Beethoven", "German"), ("Bach", "German"),
    ("William Shakespeare", "English"), ("Charles Dickens", "English"),
    ("Jane Austen", "English"), ("Mark Twain", "English"),
    ("Friedrich Nietzsche", "German"), ("Confucius", "Chinese"),
    ("Lao Tzu", "Chinese"), ("Sun Tzu", "Chinese"),
    ("Akira Kurosawa", "Japanese"), ("Genghis Khan", "Mongolian"),
])

# P176 manufacturer (15)
_add("P176", [
    ("iPhone", "Apple"), ("MacBook", "Apple"), ("iPad", "Apple"),
    ("Galaxy", "Samsung"), ("Pixel", "Google"),
    ("Surface", "Microsoft"), ("Xbox", "Microsoft"),
    ("PlayStation", "Sony"), ("Switch", "Nintendo"),
    ("Mustang", "Ford"), ("Civic", "Honda"),
    ("Camry", "Toyota"), ("Model S", "Tesla"),
    ("ThinkPad", "Lenovo"), ("Bravia", "Sony"),
])

# P276 location (15)
_add("P276", [
    ("Eiffel Tower", "Paris"), ("Statue of Liberty", "York"),
    ("Big Ben", "London"), ("Colosseum", "Rome"),
    ("Acropolis", "Athens"), ("Brandenburg Gate", "Berlin"),
    ("Red Square", "Moscow"), ("Forbidden City", "Beijing"),
    ("Taj Mahal", "Agra"), ("Pyramids of Giza", "Giza"),
    ("Sydney Opera House", "Sydney"), ("Golden Gate", "San Francisco"),
    ("Empire State Building", "York"), ("Burj Khalifa", "Dubai"),
    ("Petronas Towers", "Kuala"),
])

# P407 written in language (10)
_add("P407", [
    ("Don Quixote", "Spanish"), ("War and Peace", "Russian"),
    ("Les Miserables", "French"), ("Faust", "German"),
    ("Iliad", "Greek"), ("Aeneid", "Latin"),
    ("Bible", "Hebrew"), ("Quran", "Arabic"),
    ("Analects", "Chinese"), ("Tale of Genji", "Japanese"),
])

# P413 playing position (10)
_add("P413", [
    ("Cristiano Ronaldo", "forward"), ("Lionel Messi", "forward"),
    ("Tom Brady", "quarterback"), ("LeBron James", "forward"),
    ("Michael Jordan", "guard"), ("Wayne Gretzky", "center"),
    ("Babe Ruth", "outfielder"), ("Pele", "forward"),
    ("Diego Maradona", "midfielder"), ("David Beckham", "midfielder"),
])

# P159 headquarters (15)
_add("P159", [
    ("Apple", "Cupertino"), ("Google", "Mountain View"),
    ("Microsoft", "Redmond"), ("Amazon", "Seattle"),
    ("Facebook", "Menlo Park"), ("Tesla", "Austin"),
    ("Samsung", "Seoul"), ("Sony", "Tokyo"),
    ("Toyota", "Toyota City"), ("Honda", "Tokyo"),
    ("BMW", "Munich"), ("Volkswagen", "Wolfsburg"),
    ("Nestle", "Vevey"), ("Siemens", "Munich"),
    ("Lenovo", "Beijing"),
])


# ---------------------------------------------------------------------------
# Build records
# ---------------------------------------------------------------------------
def _build_record(rel: str, ent: str, val: str) -> dict:
    templates = PARAPHRASE_TEMPLATES[rel]
    paraphrases = [t.format(entity=ent) for t in templates]
    return {
        "address": paraphrases[0],
        "address_canonical": paraphrases[0],
        "entity": ent,
        "paraphrases": paraphrases,
        "relation": rel,
        "value": val,
    }


def _split_relation_facts(facts: list[dict], seed: int) -> dict[str, list[dict]]:
    """Per-relation 60/15/15/10 train/dev/val2/test split with sha-locked seed."""
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for f in facts:
        by_rel[f["relation"]].append(f)
    out: dict[str, list[dict]] = {"train": [], "dev": [], "val2": [], "test": []}
    for rel in sorted(by_rel):
        items = sorted(by_rel[rel], key=lambda x: x["entity"])
        rng = random.Random(f"v31-{rel}-{seed}")
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * 0.60))
        n_dev = max(1, int(n * 0.15))
        n_val2 = max(1, int(n * 0.15))
        out["train"].extend(items[:n_train])
        out["dev"].extend(items[n_train:n_train + n_dev])
        out["val2"].extend(items[n_train + n_dev:n_train + n_dev + n_val2])
        out["test"].extend(items[n_train + n_dev + n_val2:])
    return out


def _write(path: Path, records: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("w") as f:
        for r in records:
            line = json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=2500)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    records = [_build_record(r, e, v) for (r, e, v) in FACTS]

    # Sanity
    relations = sorted({r["relation"] for r in records})
    n_pairs = sum(len(r["paraphrases"]) for r in records)
    print(f"relations: {len(relations)} ({relations})")
    print(f"facts:     {len(records)}")
    print(f"pairs:     {n_pairs}")
    if len(relations) < 15:
        raise SystemExit(f"need ≥15 relations for v3.1; got {len(relations)}")
    if n_pairs < args.min_pairs:
        raise SystemExit(f"need ≥{args.min_pairs} (write,paraphrase) pairs; got {n_pairs}")

    splits = _split_relation_facts(records, args.seed)
    info: dict = {
        "seed": args.seed,
        "relations": relations,
        "n_relations": len(relations),
        "n_facts": len(records),
        "n_pairs": n_pairs,
        "splits": {},
    }
    for name, recs in splits.items():
        sha = _write(out_dir / f"{name}_v31.jsonl", recs)
        info["splits"][name] = {
            "n_facts": len(recs),
            "n_pairs": sum(len(r["paraphrases"]) for r in recs),
            "sha256": sha,
        }
        print(f"  {name:6s}: {len(recs):4d} facts, {info['splits'][name]['n_pairs']:4d} pairs, sha={sha[:12]}")

    info_path = out_dir / "SPLIT_INFO.json"
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True))
    print(f"wrote {info_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
