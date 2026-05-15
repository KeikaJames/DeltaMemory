"""Exp35b — 01: Independent paraphrases via Ollama (D3 audit).

Generates `independent_paraphrases[2]` for the TEST split only (1500 facts).
Uses qwen3-coder:30b through the Ollama HTTP API.

Caveat: qwen3-coder is in the Qwen3 architecture family. It is trained on a
substantially code-heavy corpus that is distributionally different from
Qwen3-4B-Instruct-2507's training, but it shares the tokenizer. We document
this as a partial-independence audit. A future run with a non-Qwen LM is
listed as follow-up work. (See deviations.md.)

Output: exp35b_memit_bank/data/test_independent_paraphrases.json
        {fact_id: [paraphrase1, paraphrase2]}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

PROMPT_TEMPLATE = (
    "Rewrite this prompt-completion pair as a SHORT cloze-style sentence "
    "that ends right before the answer, in DIFFERENT wording. The answer word "
    "must NOT appear in your output. Output ONLY the rewritten prompt (10-18 "
    "words), no quotes, no preamble.\n"
    "Original prompt: {prompt}\n"
    "Answer (must NOT appear): {answer}\n"
    "Example transformation: 'The mother tongue of Danielle Darrieux is' / answer 'French' -> 'Danielle Darrieux is a native speaker of'\n"
    "Rewritten prompt:"
)

PROMPT_TEMPLATE_2 = (
    "Generate a SECOND, structurally different cloze prompt for the same "
    "fact. Same constraint: answer must NOT appear. Different wording from "
    "any prior paraphrase. Output ONLY the prompt (10-18 words), no quotes.\n"
    "Original prompt: {prompt}\n"
    "Answer (must NOT appear): {answer}\n"
    "Rewritten prompt:"
)


def ollama_call(model, prompt, timeout=30):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({
            "model": model, "prompt": prompt, "stream": False,
            "options": {"num_predict": 60, "temperature": 0.7}
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d.get("response", "").strip()


def clean(text):
    text = text.strip().strip('"').strip("'").strip()
    # Drop trailing junk if model added a second sentence
    for sep in ["\n", "  "]:
        if sep in text:
            text = text.split(sep)[0].strip()
    return text


def fact_to_inputs(f):
    return {
        "prompt": f["prompt"].format(f["subject"]).strip(),
        "answer": f["target_true"].strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-coder:30b")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-paraphrases", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else DATA / f"{args.split}_independent_paraphrases.json"
    facts = json.load(open(DATA / "splits" / f"{args.split}.json"))
    print(f"[load] {len(facts)} facts in {args.split}", flush=True)

    existing = {}
    if args.resume and out_path.exists():
        existing = json.load(open(out_path))
        print(f"[resume] {len(existing)} already done", flush=True)

    t0 = time.time()
    n_done = 0
    n_skip = 0
    for i, f in enumerate(facts):
        if f["id"] in existing and len(existing[f["id"]]) >= args.n_paraphrases:
            n_skip += 1
            continue
        stmt = fact_to_inputs(f)
        paras = []
        for j in range(args.n_paraphrases):
            tmpl = PROMPT_TEMPLATE if j == 0 else PROMPT_TEMPLATE_2
            try:
                resp = ollama_call(args.model, tmpl.format(**stmt))
                p = clean(resp)
                # Reject if answer appears in paraphrase (leakage)
                if stmt["answer"].lower() in p.lower():
                    p = ""
                paras.append(p)
            except Exception as e:
                print(f"[err] fact {f['id']} para {j}: {e}", flush=True)
                paras.append("")
        existing[f["id"]] = paras
        n_done += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(1e-3, elapsed)
            eta = (len(facts) - i - 1) / max(1e-3, rate)
            print(f"  {i+1}/{len(facts)}  rate={rate:.2f}/s  eta={eta/60:.1f}m  "
                  f"sample: {paras[0][:80]!r}", flush=True)
            # checkpoint
            json.dump(existing, open(out_path, "w"), ensure_ascii=False, indent=1)

    json.dump(existing, open(out_path, "w"), ensure_ascii=False, indent=1)

    meta = {
        "model": args.model,
        "split": args.split,
        "n_facts": len(facts),
        "n_done": n_done,
        "n_skip": n_skip,
        "n_paraphrases_per_fact": args.n_paraphrases,
        "prompt_template_sha": hashlib.sha256(
            (PROMPT_TEMPLATE + PROMPT_TEMPLATE_2).encode()
        ).hexdigest()[:16],
        "elapsed_sec": time.time() - t0,
    }
    json.dump(meta, open(out_path.with_suffix(".meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
