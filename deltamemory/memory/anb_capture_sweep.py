"""Exp17 — extended capture-site resolver for ANB write prompts.

Extends ``capture_policy.resolve_capture_sites`` with the full matrix from
the master plan (§7):

  period, subject_first, subject_last, relation_first, relation_last,
  object_first, object_last, subject_relation_pair, subject_object_pair,
  full_content, all_content_sparse, paraphrase_write.

Spans are resolved by tokenizer-prefix length, exactly matching how the
write prompt was tokenized (`add_special_tokens=True` by default).  If a
span cannot be unambiguously located, the resolver returns ``None`` for
that site and the runner falls back to ``period`` (or skips, per phase).

This module does NOT modify ``capture_policy.py`` — Exp17 lives entirely
in this new file.  ``write_fact`` already accepts an explicit
``capture_pos`` argument, so the runner takes the integer position returned
here and passes it directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


SiteName = str  # see VALID_SITES below

VALID_SITES = (
    "period",
    "subject_first",
    "subject_last",
    "relation_first",
    "relation_last",
    "object_first",
    "object_last",
    "subject_relation_pair",
    "subject_object_pair",
    "full_content",
    "all_content_sparse",
    "paraphrase_write",
)


@dataclass(frozen=True)
class CaptureSpec:
    """A single capture instruction: which token position(s) to write at.

    Attributes:
        site: human-readable site name (one of VALID_SITES).
        token_positions: list of zero-indexed token positions in the
            tokenized write prompt to capture at.  Single-token sites have
            length 1; multi-token sites (pairs / full_content) emit
            multiple positions, each producing one bank slot with role
            suffix ``@site@i``.
    """
    site: SiteName
    token_positions: list[int]


def _tokenize_len(tokenizer, text: str, *, add_special: bool) -> int:
    ids = tokenizer(text, return_tensors=None, add_special_tokens=add_special)
    # transformers' BatchEncoding is dict-like but not a `dict` subclass,
    # so `isinstance(ids, dict)` is False and `len(ids)` returns the
    # number of keys. Probe for the `input_ids` field explicitly.
    if hasattr(ids, "get"):
        try:
            arr = ids.get("input_ids", None)
        except TypeError:
            arr = None
        if arr is None and "input_ids" in ids:
            arr = ids["input_ids"]
        if arr is not None:
            return len(arr)
    return len(ids)


def _span_token_range(
    tokenizer,
    write_prompt: str,
    span: str,
    *,
    add_special: bool = True,
) -> Optional[tuple[int, int]]:
    """Find the half-open token range [start, end) of ``span`` inside ``write_prompt``.

    Returns ``None`` if the span text is not found.
    Resolution is by prefix-length tokenization, which exactly preserves the
    tokenizer's own segmentation regardless of the surface form of the span
    (handles BPE / SentencePiece merges).
    """
    if not span:
        return None
    # Locate span char-wise.  Try literal first, then a relaxed strip-match.
    idx = write_prompt.find(span)
    if idx < 0:
        idx = write_prompt.find(span.strip())
        if idx < 0:
            return None
    # BPE / SentencePiece tokenizers (e.g. GPT-2, Llama) typically attach the
    # leading whitespace to the following token. If the character before the
    # span is whitespace, include that whitespace in the "inclusive" prefix
    # and drop it from the "prefix" so the token-count diff captures the
    # whole word the tokenizer actually emits.
    prefix_end = idx
    while prefix_end > 0 and write_prompt[prefix_end - 1] == " ":
        prefix_end -= 1
    prefix = write_prompt[:prefix_end]
    inclusive_prefix = write_prompt[:idx + len(span)]
    n_prefix = _tokenize_len(tokenizer, prefix, add_special=add_special)
    n_inclusive = _tokenize_len(tokenizer, inclusive_prefix, add_special=add_special)
    # The added tokens are [n_prefix, n_inclusive).  If add_special_tokens=True,
    # the BOS that prefix and inclusive share is counted in both, cancelling out.
    if n_inclusive <= n_prefix:
        return None
    return (n_prefix, n_inclusive)


def resolve_extended_capture(
    *,
    site: SiteName,
    write_prompt: str,
    subject: str,
    relation_phrase: Optional[str],
    object_str: str,
    tokenizer,
    attention_mask_row,
    add_special_tokens: bool = True,
) -> Optional[CaptureSpec]:
    """Return a CaptureSpec for the requested site, or ``None`` if unresolvable.

    Args:
        site: one of VALID_SITES.
        write_prompt: the full write prompt string.
        subject: subject span (CounterFact ``subject``).
        relation_phrase: relation template stripped of the subject placeholder
            (e.g. for prompt ``"{} is located in"`` → ``"is located in"``).
        object_str: target_new (or target_true) span being asserted.
        tokenizer: HF tokenizer; tokenization MUST match how the runner
            encodes the write prompt downstream.
        attention_mask_row: 1-D attention-mask tensor for the row (used for
            the ``period`` fallback).
        add_special_tokens: must match the runner.
    """
    if site not in VALID_SITES:
        raise ValueError(f"unknown capture site: {site!r}")

    period_pos = int(attention_mask_row.sum().item()) - 1

    if site == "period":
        return CaptureSpec(site, [period_pos])

    subj_range = _span_token_range(tokenizer, write_prompt, subject,
                                   add_special=add_special_tokens) if subject else None
    rel_range = (
        _span_token_range(tokenizer, write_prompt, relation_phrase,
                          add_special=add_special_tokens)
        if relation_phrase else None
    )
    obj_range = _span_token_range(tokenizer, write_prompt, object_str,
                                  add_special=add_special_tokens) if object_str else None

    def _pick(r, which: str) -> Optional[int]:
        if r is None:
            return None
        s, e = r
        if e <= s:
            return None
        # Clamp into the real-tokens region.
        if which == "first":
            p = s
        else:
            p = e - 1
        if p < 0 or p > period_pos:
            return None
        return p

    if site == "subject_first":
        p = _pick(subj_range, "first")
        return CaptureSpec(site, [p]) if p is not None else None
    if site == "subject_last":
        p = _pick(subj_range, "last")
        return CaptureSpec(site, [p]) if p is not None else None
    if site == "relation_first":
        p = _pick(rel_range, "first")
        return CaptureSpec(site, [p]) if p is not None else None
    if site == "relation_last":
        p = _pick(rel_range, "last")
        return CaptureSpec(site, [p]) if p is not None else None
    if site == "object_first":
        p = _pick(obj_range, "first")
        return CaptureSpec(site, [p]) if p is not None else None
    if site == "object_last":
        p = _pick(obj_range, "last")
        return CaptureSpec(site, [p]) if p is not None else None

    if site == "subject_relation_pair":
        ps = [_pick(subj_range, "last"), _pick(rel_range, "last")]
        ps = [p for p in ps if p is not None]
        return CaptureSpec(site, ps) if len(ps) == 2 else None

    if site == "subject_object_pair":
        ps = [_pick(subj_range, "last"), _pick(obj_range, "last")]
        ps = [p for p in ps if p is not None]
        return CaptureSpec(site, ps) if len(ps) == 2 else None

    if site == "full_content":
        ps = [
            _pick(subj_range, "last"),
            _pick(rel_range, "last"),
            _pick(obj_range, "last"),
        ]
        ps = [p for p in ps if p is not None]
        return CaptureSpec(site, ps) if ps else None

    if site == "all_content_sparse":
        # Every token inside any of the three spans, deduplicated, sorted.
        positions: set[int] = set()
        for rng in (subj_range, rel_range, obj_range):
            if rng is None:
                continue
            s, e = rng
            for i in range(s, e):
                if 0 <= i <= period_pos:
                    positions.add(i)
        if not positions:
            return None
        return CaptureSpec(site, sorted(positions))

    if site == "paraphrase_write":
        # Caller is responsible for swapping the write_prompt string; the
        # capture position here defaults to the period of that paraphrased
        # prompt, which is the same convention as the standard path.
        return CaptureSpec(site, [period_pos])

    return None


def derive_relation_phrase(prompt_template: str) -> Optional[str]:
    """Strip the subject placeholder from a CounterFact prompt template.

    Example: ``"{} is located in"`` → ``"is located in"``.
    Returns ``None`` if no placeholder is present.
    """
    pt = (prompt_template or "").strip()
    if pt.startswith("{}"):
        return pt[2:].strip().rstrip(",") or None
    if "{}" in pt:
        return re.sub(r"\s*\{\}\s*", " ", pt).strip()
    return None


__all__ = [
    "CaptureSpec", "VALID_SITES",
    "resolve_extended_capture", "derive_relation_phrase",
]
