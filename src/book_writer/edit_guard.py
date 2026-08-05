"""
Phase 5.2 — The Edit Guard.

The Editor is the last thing to touch the text before it becomes the book,
and nobody reviews the Editor. These checks are what make it safe to run
unattended across 150 sections. No LLM call.
"""

import re
from typing import List


def verify_edit(
    raw: str, edited: str, masked_blocks: List[str], taught_terms: List[str]
) -> List[str]:
    issues: List[str] = []

    if any(f"[[CODE_BLOCK_{i}]]" not in edited for i in range(len(masked_blocks))):
        issues.append("dropped or renamed a code block placeholder")

    ratio = len(edited.split()) / max(len(raw.split()), 1)
    if not 0.75 <= ratio <= 1.15:
        issues.append(f"length changed by {abs(1 - ratio):.0%} — expected ±15%")

    for term in taught_terms:
        if term and term.lower() not in edited.lower():
            issues.append(f"concept '{term}' present in draft, absent after edit")

    for n in set(re.findall(r"\b\d[\d,.]*\b", raw)) - set(
        re.findall(r"\b\d[\d,.]*\b", edited)
    ):
        issues.append(f"numeric value {n} disappeared")

    return issues
