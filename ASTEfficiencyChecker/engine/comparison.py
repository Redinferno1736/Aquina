"""
engine/comparison.py — compares an estimated complexity against a matched
dataset problem's known-optimal complexity, and produces a verdict.
"""

from __future__ import annotations
from dataclasses import dataclass
import re

# Rough ordering of common complexity classes, small -> large, used only to
# rank *known, parsed* classes against each other — not a general complexity
# parser/prover.
_ORDER = [
    "O(1)", "O(log n)", "O(sqrt n)", "O(n)", "O(n log n)",
    "O(n^2)", "O(n^3)", "O(2^n)", "O(n!)",
]


@dataclass
class ComparisonVerdict:
    verdict: str  # "optimal" | "suboptimal" | "unknown"
    explanation: str


def _normalize(cx: str) -> str | None:
    """Extract a known complexity class from free text by looking for the
    *first* O(...) expression and matching its inner content exactly (after
    stripping spaces/case) against a known class — not a substring search,
    since e.g. 'n' is a substring of both 'n^2' and 'nlogn' and would
    otherwise misclassify them."""
    match = re.search(r"O\(([^)]*)\)", cx, flags=re.IGNORECASE)
    if not match:
        return None
    inner = match.group(1).replace(" ", "").lower()
    for canon in _ORDER:
        bare = canon.replace("O(", "").replace(")", "").replace(" ", "").lower()
        if bare == inner:
            return canon
    return None


def compare_complexity(estimated_time: str, optimal_time: str) -> ComparisonVerdict:
    est = _normalize(estimated_time)
    opt = _normalize(optimal_time)
    if est is None or opt is None or optimal_time.lower() == "unknown":
        return ComparisonVerdict(
            "unknown",
            "Could not confidently classify one or both complexities into a standard class "
            "for direct comparison — reported estimate should still be read on its own terms.",
        )
    if est == opt:
        return ComparisonVerdict("optimal", f"Estimated complexity {est} matches the known-optimal {opt}.")
    ei, oi = _ORDER.index(est), _ORDER.index(opt)
    if ei > oi:
        return ComparisonVerdict(
            "suboptimal",
            f"Estimated complexity {est} is worse than the known-optimal {opt} for this problem — "
            "there is likely a more efficient approach.",
        )
    return ComparisonVerdict(
        "better-than-recorded-optimal (verify)",
        f"Estimated complexity {est} appears better than the recorded optimal {opt} — this is "
        "unusual; double check the estimate and the dataset entry rather than trusting it outright.",
    )
