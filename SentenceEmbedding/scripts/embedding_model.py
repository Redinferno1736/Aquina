"""
embedding_model.py

Loads the SentenceTransformer model exactly once and exposes it (and a
convenience embedding function) for every other module to reuse.
Nothing else in this project should instantiate SentenceTransformer
directly — always import `model` or `embed_texts` from here.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME
from logger import logger

logger.info("Loading embedding model '%s'...", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
logger.info("Embedding model loaded successfully.")


def embed_texts(texts: list[str], normalize: bool = True) -> np.ndarray:
    """
    Encode a list of texts into a 2D float32 numpy array of embeddings.

    Parameters
    ----------
    texts:
        The texts to embed. An empty list returns an empty array.
    normalize:
        If True, L2-normalizes each embedding so inner-product search
        is equivalent to cosine similarity.

    Returns
    -------
    A numpy array of shape (len(texts), embedding_dim), dtype float32.
    """
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype="float32")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    )
    return embeddings.astype("float32")