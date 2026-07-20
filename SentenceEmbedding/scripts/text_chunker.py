"""
text_chunker.py

Paragraph-based chunker for plain text files. Paragraphs (separated by
blank lines) are merged into chunks of approximately CHUNK_TARGET_WORDS
words, carrying roughly CHUNK_OVERLAP_WORDS words of trailing context
from one chunk into the start of the next so semantic continuity
survives chunk boundaries.

This module also serves as the fallback chunker for code_chunker.py and
json_chunker.py when their structured parsing fails.
"""

from __future__ import annotations

from typing import Any

from config import CHUNK_OVERLAP_WORDS, CHUNK_TARGET_WORDS
from logger import logger


def _split_into_paragraphs(content: str) -> list[str]:
    """Split raw text into paragraphs separated by one or more blank lines."""
    raw_paragraphs = content.split("\n\n")
    paragraphs: list[str] = []
    for raw in raw_paragraphs:
        stripped = raw.strip()
        if stripped:
            paragraphs.append(stripped)
    return paragraphs


def _words(text: str) -> list[str]:
    return text.split()


def _merge_paragraphs_with_overlap(
    paragraphs: list[str],
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Merge paragraphs into chunks of approximately `target_words` words.

    Once a chunk reaches the target size it is flushed, and the next
    chunk is seeded with the last `overlap_words` words of the flushed
    chunk so context carries across the boundary. A final chunk is only
    emitted if new paragraph content was added after the last flush.
    """
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    new_words_since_flush = 0

    for paragraph in paragraphs:
        paragraph_words = _words(paragraph)
        current_words.extend(paragraph_words)
        new_words_since_flush += len(paragraph_words)

        if len(current_words) >= target_words:
            chunks.append(" ".join(current_words))
            overlap = current_words[-overlap_words:] if overlap_words > 0 else []
            current_words = list(overlap)
            new_words_since_flush = 0

    if new_words_since_flush > 0:
        chunks.append(" ".join(current_words))

    return chunks


def _build_chunk(
    filename: str,
    filepath: str,
    extension: str,
    chunk_id: int,
    content: str,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "filepath": filepath,
        "extension": extension,
        "chunk_id": chunk_id,
        "content": content,
    }


def chunk_plain_text(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Chunk a plain text file into overlapping, paragraph-aligned chunks
    of approximately CHUNK_TARGET_WORDS words each.

    Parameters
    ----------
    file_data:
        Dictionary produced by `extract_content`, containing at least
        `filename`, `filepath`, `extension`, and `content`.

    Returns
    -------
    A list of chunk dictionaries preserving the original metadata and
    adding a sequential `chunk_id` starting at 1.
    """
    filename = file_data.get("filename", "")
    filepath = file_data.get("filepath", "")
    extension = file_data.get("extension", "")
    content = file_data.get("content", "") or ""

    if not content.strip():
        logger.warning("Empty text content for %s; skipping chunking.", filepath)
        return []

    paragraphs = _split_into_paragraphs(content)
    if not paragraphs:
        logger.warning("No paragraphs found in %s; skipping chunking.", filepath)
        return []

    merged_chunks = _merge_paragraphs_with_overlap(paragraphs)

    chunks = [
        _build_chunk(filename, filepath, extension, chunk_id, chunk_content)
        for chunk_id, chunk_content in enumerate(merged_chunks, start=1)
    ]

    logger.debug("Chunked %s into %d chunk(s) [text].", filepath, len(chunks))
    return chunks