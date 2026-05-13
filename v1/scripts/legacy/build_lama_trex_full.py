#!/usr/bin/env python3
"""Build an expanded multi-relation LAMA-TREx-style JSONL for Stage 9.

Compared with ``build_lama_curated.py`` (135 capital facts), this script
covers seven relations and explicitly tags each fact with its relation
so we can analyse per-relation transfer.

Relations covered (P-codes from Wikidata / LAMA TREx):
- P36   capital                     (country -> capital)
- P19   place_of_birth              (person -> city)
- P39   position_held / occupation  (person -> role)
- P101  field_of_work               (person -> field)
- P140  religion                    (person -> religion)
- P641  sport                       (athlete -> sport)
- P937  work_location               (person -> city)

Each line:
    {"relation": "P36", "address": "...", "value": "Paris"}

We keep BOTH single- and multi-token answers; downstream scripts can
filter via ``--multi-token-max``.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "scripts" / "data" / "lama_trex_full.jsonl"


# (relation, template, answer)
FACTS: list[tuple[str, str, str]] = []

# P36 — capital
for country, capital in [
    ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
    ("Italy", "Rome"), ("Spain", "Madrid"), ("Russia", "Moscow"),
    ("China", "Beijing"), ("Egypt", "Cairo"), ("Greece", "Athens"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Poland", "Warsaw"), ("Austria", "Vienna"),
    ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"), ("Portugal", "Lisbon"),
    ("Ireland", "Dublin"), ("Hungary", "Budapest"), ("Romania", "Bucharest"),
    ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
    ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Israel", "Jerusalem"),
    ("Jordan", "Amman"), ("India", "Delhi"), ("Pakistan", "Islamabad"),
    ("Bangladesh", "Dhaka"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
    ("Cambodia", "Phnom"), ("Laos", "Vientiane"), ("Myanmar", "Naypyidaw"),
    ("Philippines", "Manila"), ("Indonesia", "Jakarta"), ("Malaysia", "Kuala"),
    ("Singapore", "Singapore"), ("Mongolia", "Ulaanbaatar"), ("Kazakhstan", "Astana"),
    ("Uzbekistan", "Tashkent"), ("Afghanistan", "Kabul"), ("Nepal", "Kathmandu"),
    ("Korea", "Seoul"), ("Australia", "Canberra"), ("Argentina", "Buenos"),
    ("Brazil", "Brasilia"), ("Chile", "Santiago"), ("Colombia", "Bogota"),
    ("Peru", "Lima"), ("Venezuela", "Caracas"), ("Ecuador", "Quito"),
    ("Bolivia", "La Paz"), ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion"),
    ("Cuba", "Havana"), ("Mexico", "Mexico"), ("Canada", "Ottawa"),
    ("Kenya", "Nairobi"), ("Ethiopia", "Addis"), ("Nigeria", "Abuja"),
    ("Ghana", "Accra"), ("Senegal", "Dakar"), ("Morocco", "Rabat"),
    ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Libya", "Tripoli"),
    ("Sudan", "Khartoum"), ("Tanzania", "Dodoma"), ("Uganda", "Kampala"),
    ("Zambia", "Lusaka"), ("Zimbabwe", "Harare"), ("Botswana", "Gaborone"),
]:
    FACTS.append(("P36", f"The capital of {country} is", capital))

# P19 — place_of_birth (well-known, often single-token first names)
for person, city in [
    ("Albert Einstein was born in", "Ulm"),
    ("Marie Curie was born in", "Warsaw"),
    ("Isaac Newton was born in", "Woolsthorpe"),
    ("Charles Darwin was born in", "Shrewsbury"),
    ("Sigmund Freud was born in", "Freiberg"),
    ("Galileo Galilei was born in", "Pisa"),
    ("Nikola Tesla was born in", "Smiljan"),
    ("Frida Kahlo was born in", "Coyoacan"),
    ("Mahatma Gandhi was born in", "Porbandar"),
    ("Nelson Mandela was born in", "Mvezo"),
    ("Winston Churchill was born in", "Blenheim"),
    ("Ernest Hemingway was born in", "Oak"),
    ("Pablo Picasso was born in", "Malaga"),
    ("Salvador Dali was born in", "Figueres"),
    ("Vincent van Gogh was born in", "Zundert"),
    ("Claude Monet was born in", "Paris"),
    ("Edgar Allan Poe was born in", "Boston"),
    ("Mark Twain was born in", "Florida"),
    ("Steve Jobs was born in", "San Francisco"),
    ("Bill Gates was born in", "Seattle"),
]:
    FACTS.append(("P19", person, city))

# P101 — field_of_work
for person, field in [
    ("Albert Einstein worked in the field of", "physics"),
    ("Marie Curie worked in the field of", "chemistry"),
    ("Isaac Newton worked in the field of", "mathematics"),
    ("Charles Darwin worked in the field of", "biology"),
    ("Sigmund Freud worked in the field of", "psychology"),
    ("Carl Jung worked in the field of", "psychology"),
    ("Niels Bohr worked in the field of", "physics"),
    ("Ada Lovelace worked in the field of", "computing"),
    ("Alan Turing worked in the field of", "computing"),
    ("John von Neumann worked in the field of", "mathematics"),
    ("Linus Pauling worked in the field of", "chemistry"),
    ("Richard Feynman worked in the field of", "physics"),
    ("Stephen Hawking worked in the field of", "physics"),
    ("Carl Sagan worked in the field of", "astronomy"),
    ("Jane Goodall worked in the field of", "biology"),
    ("Rosalind Franklin worked in the field of", "biology"),
    ("Noam Chomsky worked in the field of", "linguistics"),
    ("Friedrich Nietzsche worked in the field of", "philosophy"),
    ("Karl Marx worked in the field of", "economics"),
    ("Adam Smith worked in the field of", "economics"),
]:
    FACTS.append(("P101", person, field))

# P641 — sport
for athlete, sport in [
    ("Michael Jordan plays the sport of", "basketball"),
    ("LeBron James plays the sport of", "basketball"),
    ("Kobe Bryant played the sport of", "basketball"),
    ("Lionel Messi plays the sport of", "soccer"),
    ("Cristiano Ronaldo plays the sport of", "soccer"),
    ("Pele played the sport of", "soccer"),
    ("Diego Maradona played the sport of", "soccer"),
    ("Roger Federer plays the sport of", "tennis"),
    ("Serena Williams plays the sport of", "tennis"),
    ("Rafael Nadal plays the sport of", "tennis"),
    ("Tiger Woods plays the sport of", "golf"),
    ("Usain Bolt competes in the sport of", "athletics"),
    ("Michael Phelps competes in the sport of", "swimming"),
    ("Simone Biles competes in the sport of", "gymnastics"),
    ("Wayne Gretzky played the sport of", "hockey"),
    ("Tom Brady plays the sport of", "football"),
    ("Babe Ruth played the sport of", "baseball"),
    ("Muhammad Ali competed in the sport of", "boxing"),
    ("Floyd Mayweather competes in the sport of", "boxing"),
    ("Conor McGregor competes in the sport of", "MMA"),
]:
    FACTS.append(("P641", athlete, sport))

# P140 — religion (delicate; only well-documented historical figures)
for person, rel in [
    ("Mahatma Gandhi practised the religion of", "Hinduism"),
    ("Dalai Lama practises the religion of", "Buddhism"),
    ("Pope Francis practises the religion of", "Christianity"),
    ("Mother Teresa practised the religion of", "Christianity"),
    ("Maimonides practised the religion of", "Judaism"),
    ("Rumi practised the religion of", "Islam"),
    ("Confucius taught", "Confucianism"),
    ("Lao Tzu taught", "Taoism"),
    ("Siddhartha Gautama founded", "Buddhism"),
    ("Guru Nanak founded", "Sikhism"),
]:
    FACTS.append(("P140", person, rel))

# P39 — occupation
for person, occ in [
    ("Albert Einstein worked as a", "physicist"),
    ("Marie Curie worked as a", "chemist"),
    ("Charles Darwin worked as a", "biologist"),
    ("Sigmund Freud worked as a", "psychologist"),
    ("William Shakespeare worked as a", "playwright"),
    ("Leonardo da Vinci worked as a", "painter"),
    ("Vincent van Gogh worked as a", "painter"),
    ("Pablo Picasso worked as a", "painter"),
    ("Wolfgang Amadeus Mozart worked as a", "composer"),
    ("Johann Sebastian Bach worked as a", "composer"),
    ("Ludwig van Beethoven worked as a", "composer"),
    ("Frederic Chopin worked as a", "composer"),
    ("Steven Spielberg works as a", "director"),
    ("Martin Scorsese works as a", "director"),
    ("Christopher Nolan works as a", "director"),
    ("Stephen King works as a", "author"),
    ("J K Rowling works as an", "author"),
    ("Ernest Hemingway worked as an", "author"),
    ("Jane Austen worked as an", "author"),
    ("Charles Dickens worked as an", "author"),
]:
    FACTS.append(("P39", person, occ))

# P937 — work_location
for person, city in [
    ("Albert Einstein worked in", "Princeton"),
    ("Charles Darwin worked in", "London"),
    ("Sigmund Freud worked in", "Vienna"),
    ("Marie Curie worked in", "Paris"),
    ("Isaac Newton worked in", "Cambridge"),
    ("Galileo Galilei worked in", "Padua"),
    ("Pablo Picasso worked in", "Paris"),
    ("Vincent van Gogh worked in", "Arles"),
    ("Claude Monet worked in", "Giverny"),
    ("Leonardo da Vinci worked in", "Florence"),
    ("Michelangelo worked in", "Rome"),
    ("Raphael worked in", "Rome"),
    ("Niels Bohr worked in", "Copenhagen"),
    ("Werner Heisenberg worked in", "Munich"),
    ("Erwin Schrodinger worked in", "Dublin"),
    ("Linus Pauling worked in", "Pasadena"),
    ("Richard Feynman worked in", "Pasadena"),
    ("Carl Sagan worked in", "Ithaca"),
]:
    FACTS.append(("P937", person, city))


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_rel: dict[str, int] = {}
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rel, addr, val in FACTS:
            f.write(json.dumps({"relation": rel, "address": addr, "value": val}, ensure_ascii=False) + "\n")
            by_rel[rel] = by_rel.get(rel, 0) + 1
    print(f"wrote {len(FACTS)} facts -> {OUT_PATH}")
    for k, v in sorted(by_rel.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
