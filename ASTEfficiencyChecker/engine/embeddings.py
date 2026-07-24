"""
engine/embeddings.py — plug-in point for a Sentence Embedding model.

ASTEfficiencyChecker (this repo, Dev B / B2) does not train or ship an
embedding model — that is Dev B's B3 item on the split plan (MiniLM/SBERT).
This module just defines the interface so B3's model can be wired in later
with a single call, and provides a trivial fallback (token-overlap
similarity) so the matcher in engine/matcher.py still works end-to-end
today without B3 existing yet.
"""

from __future__ import annotations
from typing import Callable, Optional
import math
import re

_embedder: Optional[Callable[[str], list[float]]] = None


def set_embedder(fn: Callable[[str], list[float]]) -> None:
    """Wire in a real embedding model: fn(text) -> vector.
    Example once B3 exists:
        from my_sentence_embedding_model import embed
        engine.embeddings.set_embedder(embed)
    """
    global _embedder
    _embedder = fn


def has_real_embedder() -> bool:
    return _embedder is not None


def embed(text: str) -> list[float]:
    if _embedder is not None:
        return _embedder(text)
    return _fallback_bow_vector(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if isinstance(a, dict) or isinstance(b, dict):
        return _sparse_cosine(a, b)
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Fallback: no ML dependency, deterministic, testable today. Represents text
# as a sparse bag-of-words vector (dict[token] = count) so matcher.py works
# with zero external dependencies until a real embedder is wired in.
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}")


def _fallback_bow_vector(text: str) -> dict:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    vec: dict[str, int] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def _sparse_cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
