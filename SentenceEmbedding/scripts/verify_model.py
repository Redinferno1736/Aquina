"""
verify_model.py

Standalone sanity check confirming the embedding model loads correctly
and produces embeddings of the expected shape and similarity behavior.
Run this after any environment or dependency change to confirm the
embedding pipeline still works before running the full indexing job.
"""

from __future__ import annotations

import numpy as np

from config import EMBEDDING_DIM, MODEL_NAME
from embedding_model import embed_texts
from logger import logger


def _cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return float(np.dot(vector_a, vector_b))


def verify_model() -> None:
    logger.info("Verifying embedding model '%s'...", MODEL_NAME)

    sample_texts = [
        "Open the resume tracker notes",
        "Launch the resume tracking application",
        "What is the weather like today",
    ]

    embeddings = embed_texts(sample_texts, normalize=True)

    assert embeddings.shape == (len(sample_texts), EMBEDDING_DIM), (
        f"Unexpected embedding shape: {embeddings.shape}, "
        f"expected ({len(sample_texts)}, {EMBEDDING_DIM})"
    )
    logger.info("Embedding shape check passed: %s", embeddings.shape)

    similar_score = _cosine_similarity(embeddings[0], embeddings[1])
    unrelated_score = _cosine_similarity(embeddings[0], embeddings[2])

    logger.info(
        "Similarity(related pair) = %.4f, Similarity(unrelated pair) = %.4f",
        similar_score,
        unrelated_score,
    )

    if similar_score > unrelated_score:
        logger.info("Sanity check passed: related sentences scored more similar.")
    else:
        logger.warning(
            "Sanity check failed: unrelated pair scored higher than related pair."
        )

    logger.info("Model verification complete.")


if __name__ == "__main__":
    verify_model()