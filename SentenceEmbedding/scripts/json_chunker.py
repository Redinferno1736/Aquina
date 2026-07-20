"""
json_chunker.py

Top-level-key chunker for JSON files. Each top-level key in the JSON
object becomes its own chunk, with the key's value pretty-printed via
`json.dumps(indent=2)`.

Falls back to paragraph-based text chunking (via `text_chunker.py`) if
the file is not valid JSON, or is valid JSON but its root is not an
object.
"""

from __future__ import annotations

import json
from typing import Any

from logger import logger
from text_chunker import chunk_plain_text


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


def chunk_json(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Chunk a JSON file by its top-level keys.

    Parameters
    ----------
    file_data:
        Dictionary produced by `extract_content`, containing at least
        `filename`, `filepath`, `extension`, and `content`.

    Returns
    -------
    A list of chunk dictionaries preserving the original metadata and
    adding a sequential `chunk_id` starting at 1. Falls back to
    `chunk_plain_text` if the content is not valid JSON, or its root is
    not an object.
    """
    filename = file_data.get("filename", "")
    filepath = file_data.get("filepath", "")
    extension = file_data.get("extension", "")
    content = file_data.get("content", "") or ""

    if not content.strip():
        logger.warning("Empty JSON content for %s; skipping chunking.", filepath)
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse %s as JSON (%s); falling back to text chunking.",
            filepath,
            exc,
        )
        return chunk_plain_text(file_data)

    if not isinstance(parsed, dict):
        logger.warning(
            "%s is valid JSON but its root is not an object (got %s); "
            "falling back to text chunking.",
            filepath,
            type(parsed).__name__,
        )
        return chunk_plain_text(file_data)

    if not parsed:
        logger.warning("%s is an empty JSON object; skipping chunking.", filepath)
        return []

    chunks: list[dict[str, Any]] = []
    for chunk_id, (key, value) in enumerate(parsed.items(), start=1):
        pretty_value = json.dumps(value, indent=2, ensure_ascii=False)
        chunk_content = f'"{key}"\n\n{pretty_value}'
        chunks.append(_build_chunk(filename, filepath, extension, chunk_id, chunk_content))

    logger.debug("Chunked %s into %d chunk(s) [json].", filepath, len(chunks))
    return chunks