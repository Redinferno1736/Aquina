"""
generate_embeddings.py

Attaches an "embedding" vector to each chunk dictionary using the
shared embedding model loaded in embedding_model.py. No new
SentenceTransformer instance is created here.
"""

from __future__ import annotations

from typing import Any

from embedding_model import embed_texts
from logger import logger


def generate_embeddings(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Compute an embedding for each chunk's `content` and attach it to the
    chunk dictionary under the `embedding` key.

    Parameters
    ----------
    chunks:
        List of chunk dictionaries, each containing at least a
        `content` key.

    Returns
    -------
    The same list of chunk dictionaries, each augmented with an
    `embedding` key holding a float32 numpy array. Chunks with empty
    content are skipped and excluded from the returned list.
    """
    if not chunks:
        return []

    valid_chunks = [chunk for chunk in chunks if (chunk.get("content") or "").strip()]
    if not valid_chunks:
        logger.warning("No chunks with non-empty content to embed.")
        return []

    texts = [chunk["content"] for chunk in valid_chunks]
    vectors = embed_texts(texts, normalize=True)

    for chunk, vector in zip(valid_chunks, vectors):
        chunk["embedding"] = vector

    logger.debug("Generated %d embedding(s).", len(valid_chunks))
    return valid_chunks