"""
chunk_text.py

Dispatcher module. Routes a file's extracted content to the correct
format-specific chunker based on its extension. Contains no chunking
logic itself.
"""

from __future__ import annotations

from typing import Any

from code_chunker import chunk_code
from config import CODE_EXTENSIONS, JSON_EXTENSIONS, MARKDOWN_EXTENSIONS
from json_chunker import chunk_json
from logger import logger
from markdown_chunker import chunk_markdown
from text_chunker import chunk_plain_text


def chunk_text(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Dispatch `file_data` to the appropriate chunker based on its
    `extension` field.

    Parameters
    ----------
    file_data:
        Dictionary produced by `extract_content`, containing at least
        `filename`, `filepath`, `extension`, and `content`.

    Returns
    -------
    A list of chunk dictionaries, as produced by whichever chunker
    handled this file's extension.
    """
    extension = (file_data.get("extension") or "").lower()

    if extension in MARKDOWN_EXTENSIONS:
        return chunk_markdown(file_data)

    if extension in CODE_EXTENSIONS:
        return chunk_code(file_data)

    if extension in JSON_EXTENSIONS:
        return chunk_json(file_data)

    logger.debug(
        "No specialized chunker for extension '%s' (%s); using plain text chunker.",
        extension,
        file_data.get("filepath", ""),
    )
    return chunk_plain_text(file_data)