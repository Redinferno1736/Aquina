"""
search_engine.py

Encapsulates semantic search over the Aquina index as a reusable class.
The FAISS index, chunk metadata, and embedding model are all loaded
exactly once at construction time — SearchEngine.search() performs no
reloading, only query embedding and lookup.
"""

from __future__ import annotations

import os
import pickle
from typing import Any

import faiss

from config import INDEX_FILE, METADATA_FILE, PREVIEW_CHARS, TOP_K
from embedding_model import embed_texts
from logger import logger


class SearchEngine:
    """
    Reusable semantic search engine over the Aquina FAISS index.

    The embedding model (via embedding_model.py), FAISS index, and
    chunk metadata are all loaded once, at construction time, and held
    in memory for the lifetime of the instance. Construct one instance
    and reuse it for every query rather than creating a new one per
    search.

    Raises
    ------
    FileNotFoundError:
        If the index or metadata file does not exist at construction
        time, since there is nothing to search without them.
    """

    def __init__(
        self,
        index_path: str = INDEX_FILE,
        metadata_path: str = METADATA_FILE,
    ) -> None:
        self._index_path = index_path
        self._metadata_path = metadata_path

        self._index: faiss.Index = self._load_index()
        self._metadata: list[dict[str, Any]] = self._load_metadata()

        logger.info(
            "SearchEngine ready: %d vector(s) in index, %d metadata entry(ies).",
            self._index.ntotal,
            len(self._metadata),
        )

    def _load_index(self) -> faiss.Index:
        """Load the FAISS index from disk, failing loudly if it's missing."""
        if not os.path.isfile(self._index_path):
            raise FileNotFoundError(
                f"Index file not found at {self._index_path}. "
                "Run build_index.py before starting SearchEngine."
            )

        try:
            index = faiss.read_index(self._index_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FAISS index from {self._index_path}: {exc}"
            ) from exc

        return index

    def _load_metadata(self) -> list[dict[str, Any]]:
        """Load chunk metadata from disk, failing loudly if it's missing."""
        if not os.path.isfile(self._metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {self._metadata_path}. "
                "Run build_index.py before starting SearchEngine."
            )

        try:
            with open(self._metadata_path, "rb") as handle:
                metadata = pickle.load(handle)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load metadata from {self._metadata_path}: {exc}"
            ) from exc

        if not isinstance(metadata, list):
            raise ValueError(
                f"Metadata at {self._metadata_path} is malformed "
                f"(expected list, got {type(metadata).__name__})."
            )

        return metadata

    @staticmethod
    def _make_preview(content: str, max_chars: int = PREVIEW_CHARS) -> str:
        """Truncate chunk content to a short preview string."""
        stripped = (content or "").strip()
        if len(stripped) <= max_chars:
            return stripped
        return stripped[:max_chars].rstrip() + "..."

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        min_score: float = 0.20,
    ) -> list[dict[str, Any]]:
        """
        Run a semantic search over the loaded index.

        Parameters
        ----------
        query:
            The natural-language search query. Must be non-empty.
        top_k:
            Maximum number of results to return, before min_score
            filtering is applied.
        min_score:
            Minimum cosine similarity score a result must meet to be
            included in the returned list. Results below this
            threshold are filtered out.

        Returns
        -------
        A list of result dictionaries, sorted by descending similarity
        score, each containing:
        `rank`, `score`, `filename`, `filepath`, `chunk_id`, `preview`.
        Returns an empty list if the query is empty, the index has no
        vectors, or no result meets `min_score`.

        Raises
        ------
        ValueError:
            If `top_k` is not a positive integer.
        RuntimeError:
            If embedding the query or querying the FAISS index fails.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to SearchEngine.search().")
            return []

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        if self._index.ntotal == 0 or not self._metadata:
            logger.warning("Index is empty; nothing to search.")
            return []

        effective_top_k = min(top_k, self._index.ntotal)

        try:
            query_vector = embed_texts([query], normalize=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to embed query '{query}': {exc}") from exc

        try:
            scores, indices = self._index.search(query_vector, effective_top_k)
        except Exception as exc:
            raise RuntimeError(f"FAISS search failed for query '{query}': {exc}") from exc

        candidates: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue

            score_value = float(score)
            if score_value < min_score:
                continue

            chunk_meta = self._metadata[idx]
            candidates.append(
                {
                    "score": score_value,
                    "filename": chunk_meta.get("filename", ""),
                    "filepath": chunk_meta.get("filepath", ""),
                    "chunk_id": chunk_meta.get("chunk_id", -1),
                    "preview": self._make_preview(chunk_meta.get("content", "")),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)

        results: list[dict[str, Any]] = []
        for rank, candidate in enumerate(candidates, start=1):
            results.append({"rank": rank, **candidate})

        logger.debug(
            "Query '%s' returned %d result(s) above min_score=%.2f.",
            query,
            len(results),
            min_score,
        )

        return results