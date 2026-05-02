#!/usr/bin/env python3
"""Build a curated single-token-answer LAMA-style JSONL for Stage 8 Phase E.

Each line:
  {"address": "<question template / cloze>", "value": "<answer word>"}

We start from a hand-curated pool of factual triples (capitals, languages,
currencies, country-of-origin facts) and keep only those whose answer
tokenizes to a single token under the target tokenizer (with a leading
space). Output goes to scripts/data/lama_curated.jsonl.

Run on the box that has the tokenizer cached (Mac is fine; this script
does not need a GPU or model weights — only the tokenizer).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "scripts" / "data" / "lama_curated.jsonl"

# (template, answer) pairs. We use natural English templates and
# single-word answers commonly tokenized as one piece by SentencePiece /
# Gemma tokenizers.
RAW_FACTS: list[tuple[str, str]] = [
    # Capitals (country -> capital)
    ("The capital of France is", "Paris"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Spain is", "Madrid"),
    ("The capital of Russia is", "Moscow"),
    ("The capital of China is", "Beijing"),
    ("The capital of Egypt is", "Cairo"),
    ("The capital of India is", "Delhi"),
    ("The capital of Greece is", "Athens"),
    ("The capital of Turkey is", "Ankara"),
    ("The capital of Portugal is", "Lisbon"),
    ("The capital of Austria is", "Vienna"),
    ("The capital of Hungary is", "Budapest"),
    ("The capital of Poland is", "Warsaw"),
    ("The capital of Sweden is", "Stockholm"),
    ("The capital of Norway is", "Oslo"),
    ("The capital of Denmark is", "Copenhagen"),
    ("The capital of Finland is", "Helsinki"),
    ("The capital of Iceland is", "Reykjavik"),
    ("The capital of Ireland is", "Dublin"),
    ("The capital of Belgium is", "Brussels"),
    ("The capital of Netherlands is", "Amsterdam"),
    ("The capital of Switzerland is", "Bern"),
    ("The capital of Czechia is", "Prague"),
    ("The capital of Romania is", "Bucharest"),
    ("The capital of Bulgaria is", "Sofia"),
    ("The capital of Ukraine is", "Kyiv"),
    ("The capital of Belarus is", "Minsk"),
    ("The capital of Croatia is", "Zagreb"),
    ("The capital of Serbia is", "Belgrade"),
    ("The capital of Iran is", "Tehran"),
    ("The capital of Iraq is", "Baghdad"),
    ("The capital of Israel is", "Jerusalem"),
    ("The capital of Lebanon is", "Beirut"),
    ("The capital of Jordan is", "Amman"),
    ("The capital of Syria is", "Damascus"),
    ("The capital of Pakistan is", "Islamabad"),
    ("The capital of Bangladesh is", "Dhaka"),
    ("The capital of Thailand is", "Bangkok"),
    ("The capital of Vietnam is", "Hanoi"),
    ("The capital of Cambodia is", "Phnom"),
    ("The capital of Indonesia is", "Jakarta"),
    ("The capital of Malaysia is", "Kuala"),
    ("The capital of Singapore is", "Singapore"),
    ("The capital of Philippines is", "Manila"),
    ("The capital of Korea is", "Seoul"),
    ("The capital of Mongolia is", "Ulaanbaatar"),
    ("The capital of Afghanistan is", "Kabul"),
    ("The capital of Mexico is", "Mexico"),
    ("The capital of Cuba is", "Havana"),
    ("The capital of Brazil is", "Brasilia"),
    ("The capital of Argentina is", "Buenos"),
    ("The capital of Chile is", "Santiago"),
    ("The capital of Peru is", "Lima"),
    ("The capital of Venezuela is", "Caracas"),
    ("The capital of Colombia is", "Bogota"),
    ("The capital of Uruguay is", "Montevideo"),
    ("The capital of Bolivia is", "La"),
    ("The capital of Ecuador is", "Quito"),
    ("The capital of Canada is", "Ottawa"),
    ("The capital of Australia is", "Canberra"),
    ("The capital of Kenya is", "Nairobi"),
    ("The capital of Ethiopia is", "Addis"),
    ("The capital of Nigeria is", "Abuja"),
    ("The capital of Ghana is", "Accra"),
    ("The capital of Senegal is", "Dakar"),
    ("The capital of Morocco is", "Rabat"),
    ("The capital of Tunisia is", "Tunis"),
    ("The capital of Algeria is", "Algiers"),
    ("The capital of Libya is", "Tripoli"),

    # Continents (country -> continent)
    ("The country France is located in", "Europe"),
    ("The country Japan is located in", "Asia"),
    ("The country Brazil is located in", "Americas"),
    ("The country Egypt is located in", "Africa"),
    ("The country Australia is located in", "Oceania"),

    # Currencies (country -> currency)
    ("The currency of Japan is", "yen"),
    ("The currency of China is", "yuan"),
    ("The currency of Russia is", "ruble"),
    ("The currency of India is", "rupee"),
    ("The currency of Mexico is", "peso"),
    ("The currency of Sweden is", "krona"),
    ("The currency of Norway is", "krone"),
    ("The currency of Poland is", "zloty"),
    ("The currency of Turkey is", "lira"),
    ("The currency of Israel is", "shekel"),
    ("The currency of Korea is", "won"),
    ("The currency of Vietnam is", "dong"),
    ("The currency of Thailand is", "baht"),
    ("The currency of UK is", "pound"),
    ("The currency of US is", "dollar"),

    # Languages spoken (country -> primary language)
    ("The official language of France is", "French"),
    ("The official language of Germany is", "German"),
    ("The official language of Japan is", "Japanese"),
    ("The official language of Italy is", "Italian"),
    ("The official language of Spain is", "Spanish"),
    ("The official language of Russia is", "Russian"),
    ("The official language of China is", "Chinese"),
    ("The official language of Greece is", "Greek"),
    ("The official language of Turkey is", "Turkish"),
    ("The official language of Portugal is", "Portuguese"),
    ("The official language of Sweden is", "Swedish"),
    ("The official language of Norway is", "Norwegian"),
    ("The official language of Denmark is", "Danish"),
    ("The official language of Finland is", "Finnish"),
    ("The official language of Poland is", "Polish"),
    ("The official language of Hungary is", "Hungarian"),
    ("The official language of Romania is", "Romanian"),
    ("The official language of Vietnam is", "Vietnamese"),
    ("The official language of Thailand is", "Thai"),
    ("The official language of Indonesia is", "Indonesian"),
    ("The official language of Korea is", "Korean"),
    ("The official language of Mongolia is", "Mongolian"),
    ("The official language of Iran is", "Persian"),
    ("The official language of Egypt is", "Arabic"),
    ("The official language of Israel is", "Hebrew"),

    # Color sky / common simple facts (test that the bank, not pretrained
    # priors, is doing the work — these are still factual-ish but the
    # template + frozen base alone may also produce them; they're filtered
    # by no_memory baseline at eval time)
    ("The color of grass is", "green"),
    ("The color of the sun is", "yellow"),
    ("The color of snow is", "white"),
    ("The color of coal is", "black"),

    # Number of (X)
    ("The number of days in a week is", "seven"),
    ("The number of months in a year is", "twelve"),
    ("The number of continents is", "seven"),
    ("The number of planets is", "eight"),

    # Authors / iconic single-token surnames
    ("The author of Hamlet is", "Shakespeare"),
    ("The author of Inferno is", "Dante"),
    ("The author of Faust is", "Goethe"),
    ("The author of Iliad is", "Homer"),

    # Composer / classical
    ("The composer of the Ninth Symphony is", "Beethoven"),
    ("The composer of the Magic Flute is", "Mozart"),
    ("The composer of Bolero is", "Ravel"),

    # Element / chemistry
    ("The chemical symbol H stands for", "hydrogen"),
    ("The chemical symbol O stands for", "oxygen"),
    ("The chemical symbol N stands for", "nitrogen"),
    ("The chemical symbol C stands for", "carbon"),
    ("The chemical symbol Fe stands for", "iron"),
    ("The chemical symbol Au stands for", "gold"),
    ("The chemical symbol Ag stands for", "silver"),
    ("The chemical symbol Cu stands for", "copper"),

    # Sports
    ("The most popular ball sport in the US is", "football"),
    ("The most popular sport in India is", "cricket"),

    # Geography rivers / mountains
    ("The longest river in Egypt is", "Nile"),
    ("The longest river in Brazil is", "Amazon"),
    ("The longest river in China is", "Yangtze"),
    ("The highest mountain on Earth is", "Everest"),
]


def main() -> int:
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        print(f"transformers required: {e}", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    kept: list[dict] = []
    dropped: list[tuple[str, str, list[int]]] = []
    for tmpl, ans in RAW_FACTS:
        ids = tok.encode(" " + ans, add_special_tokens=False)
        if len(ids) == 1:
            kept.append({"address": tmpl, "value": ans})
        else:
            dropped.append((tmpl, ans, ids))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"[curate] kept {len(kept)} / {len(RAW_FACTS)}; wrote {OUT_PATH}")
    if dropped:
        print(f"[curate] dropped {len(dropped)} multi-token answers (showing first 10):")
        for tmpl, ans, ids in dropped[:10]:
            print(f"  {ans!r} -> {ids}  ({tmpl!r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
