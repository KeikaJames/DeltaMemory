"""Stage 14B / 14C — address-conditional capture policy.

The v2 baseline captures K at the last real token of the write prompt
(usually the period of ``"<address> is <value>."``). At read time, queries
phrased as ``"Q: ... A:"`` capture Q at the ``A:`` token, which lives in a
different region of K-space → recall@1 ≈ 0.003 in 13B-1.

Stage 14B replaces this with **address-conditional capture**: identify the
address span inside the write prompt and capture K at the last token of
that span. Stage 14C adds ``policy="multi"`` which captures at *both* the
address span and the period, doubling the bank slots and widening the
retrieval radius.

Policies
--------
- ``"period"``: legacy v2 behaviour (last real token).
- ``"address"``: capture at the last token of the address span.
- ``"multi"``: emit both ``"period"`` and ``"address"`` captures.

The address span is recovered from the structured fact record (the
``"address"`` field of the LAMA-TREx records, e.g. ``"The capital of
France is"``). If not provided, we fall back to a regex
``^(.+?) is .+?\\.`` over the write prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CapturePolicy = str  # "period" | "address" | "multi"

_ADDRESS_RE = re.compile(r"^(.+?)\s+is\s+.+?\.\s*$")


@dataclass(frozen=True)
class CaptureSite:
    """A single (token_position, role) site to capture during write.

    Attributes:
        token_pos: zero-indexed position within the tokenized write prompt.
        role: one of ``"period"`` or ``"address"`` (debug metadata only).
    """

    token_pos: int
    role: str


def _last_real_token_pos(attention_mask_row) -> int:
    return int(attention_mask_row.sum().item()) - 1


def _address_token_pos(
    write_prompt: str,
    address: str | None,
    tokenizer,
    add_special_tokens: bool = True,
) -> int | None:
    """Return the token position of the last token of the address span.

    Strategy:
      1. If ``address`` is provided, find ``address`` as a prefix of
         ``write_prompt`` (case-sensitive, allowing trailing whitespace),
         tokenize the prefix, and return ``len(prefix_ids) - 1``.
      2. Otherwise, regex-extract ``"X"`` from ``"X is Y."`` and recurse.
      3. If neither works, return ``None`` (caller falls back to period).
    """
    if address is None:
        m = _ADDRESS_RE.match(write_prompt)
        if not m:
            return None
        address = m.group(1)

    addr = address.strip()
    if not write_prompt.startswith(addr):
        # Permit a leading-space variant.
        if not write_prompt.lstrip().startswith(addr):
            return None
        write_prompt = write_prompt.lstrip()

    prefix_ids = tokenizer(addr, return_tensors=None, add_special_tokens=add_special_tokens)
    if isinstance(prefix_ids, dict):
        prefix_ids = prefix_ids["input_ids"]
    if not prefix_ids:
        return None
    return len(prefix_ids) - 1


def resolve_capture_sites(
    *,
    policy: CapturePolicy,
    write_prompt: str,
    address: str | None,
    tokenizer,
    attention_mask_row,
    add_special_tokens: bool = True,
) -> list[CaptureSite]:
    """Return the list of capture sites for a single write prompt.

    Args:
        policy: ``"period"``, ``"address"``, or ``"multi"``.
        write_prompt: the full write prompt string.
        address: the address span if known (preferred); otherwise the regex
            fallback is used.
        tokenizer: HF tokenizer.
        attention_mask_row: 1-D attention-mask tensor for the row.
        add_special_tokens: must match how the write prompt itself was
            tokenized so token positions line up.

    Returns:
        Non-empty list of :class:`CaptureSite`. Always at least one site
        (falls back to ``period`` if address resolution fails).
    """
    period_pos = _last_real_token_pos(attention_mask_row)

    if policy == "period":
        return [CaptureSite(period_pos, "period")]

    addr_pos = _address_token_pos(
        write_prompt=write_prompt,
        address=address,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
    )

    if policy == "address":
        if addr_pos is None or addr_pos >= period_pos:
            # Honest fallback: address pos missing or off the end -> period.
            return [CaptureSite(period_pos, "period")]
        return [CaptureSite(addr_pos, "address")]

    if policy == "multi":
        sites = [CaptureSite(period_pos, "period")]
        if addr_pos is not None and 0 <= addr_pos < period_pos:
            sites.insert(0, CaptureSite(addr_pos, "address"))
        return sites

    raise ValueError(f"unknown capture policy: {policy!r}")
