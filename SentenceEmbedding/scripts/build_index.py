"""
build_index.py

Walks DATA_DIR, extracts and chunks every non-ignored file, embeds each
chunk, and builds a FAISS inner-product index over the results. The
index and its accompanying chunk metadata are saved to disk so
search_index.py can load them without re-indexing.

Indexing continues even if an individual file fails to extract, chunk,
or embed — the failure is logged and the file is skipped.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any

import faiss
import numpy as np

from chunk_text import chunk_text
from config import (
    DATA_DIR,
    EMBEDDING_DIM,
    IGNORED_EXTENSIONS,
    IGNORED_FILES,
    IGNORED_FOLDERS,
    INDEX_FILE,
    METADATA_FILE,
)
from extract_content import extract_content
from generate_embeddings import generate_embeddings
from logger import logger


def _should_skip_folder(folder_name: str) -> bool:
    return folder_name in IGNORED_FOLDERS or folder_name.startswith(".")


def _should_skip_file(filename: str) -> bool:
    if filename in IGNORED_FILES:
        return True
    extension = os.path.splitext(filename)[1].lower()
    return extension in IGNORED_EXTENSIONS


def _collect_files(root_dir: str) -> list[str]:
    """Walk root_dir and return absolute paths of every file to be indexed."""
    collected: list[str] = []

    for current_dir, subdirs, filenames in os.walk(root_dir):
        subdirs[:] = [d for d in subdirs if not _should_skip_folder(d)]

        for filename in filenames:
            if _should_skip_file(filename):
                continue
            collected.append(os.path.join(current_dir, filename))

    return collected


def _process_file(filepath: str) -> list[dict[str, Any]]:
    """Extract, chunk, and embed a single file. Returns an empty list on any failure."""
    try:
        file_data = extract_content(filepath)
    except Exception:
        logger.exception("Failed to extract content from %s; skipping.", filepath)
        return []

    try:
        chunks = chunk_text(file_data)
    except Exception:
        logger.exception("Failed to chunk %s; skipping.", filepath)
        return []

    if not chunks:
        return []

    try:
        embedded_chunks = generate_embeddings(chunks)
    except Exception:
        logger.exception("Failed to embed chunks for %s; skipping.", filepath)
        return []

    return embedded_chunks


def build_index() -> None:
    """
    Build the FAISS index and metadata store from every eligible file
    under DATA_DIR, then persist both to disk.
    """
    start_time = time.perf_counter()

    if not os.path.isdir(DATA_DIR):
        logger.error("DATA_DIR does not exist: %s", DATA_DIR)
        return

    logger.info("Scanning %s for indexable files...", DATA_DIR)
    filepaths = _collect_files(DATA_DIR)
    logger.info("Found %d candidate file(s).", len(filepaths))

    all_vectors: list[np.ndarray] = []
    all_metadata: list[dict[str, Any]] = []

    files_indexed = 0
    files_failed = 0

    for filepath in filepaths:
        embedded_chunks = _process_file(filepath)

        if not embedded_chunks:
            files_failed += 1
            continue

        for chunk in embedded_chunks:
            all_vectors.append(chunk["embedding"])
            all_metadata.append(
                {
                    "filename": chunk["filename"],
                    "filepath": chunk["filepath"],
                    "extension": chunk["extension"],
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                }
            )

        files_indexed += 1

    if not all_vectors:
        logger.warning("No embeddings were generated; index was not built.")
        return

    vector_matrix = np.vstack(all_vectors).astype("float32")

    logger.info("Building FAISS index for %d chunk(s)...", len(all_metadata))
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vector_matrix)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as handle:
        pickle.dump(all_metadata, handle)

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Indexing complete: %d file(s) indexed, %d file(s) failed/skipped, "
        "%d chunk(s) total, %.2f seconds elapsed.",
        files_indexed,
        files_failed,
        len(all_metadata),
        elapsed,
    )
    logger.info("Index saved to %s", INDEX_FILE)
    logger.info("Metadata saved to %s", METADATA_FILE)


if __name__ == "__main__":
    build_index()