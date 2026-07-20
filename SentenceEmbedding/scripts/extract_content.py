"""
extract_content.py

Reads a single file from disk and returns its metadata plus raw text
content in the standard shape used throughout the pipeline:

    {
        "filename": str,
        "filepath": str,
        "extension": str,
        "content": str,
    }
"""

from __future__ import annotations

import os
from typing import Any

from logger import logger

_ENCODING_FALLBACKS = ("utf-8", "utf-8-sig", "latin-1")


def _read_file_text(filepath: str) -> str:
    """Read a file's text content, trying a small set of encodings in order."""
    last_error: Exception | None = None

    for encoding in _ENCODING_FALLBACKS:
        try:
            with open(filepath, "r", encoding=encoding) as handle:
                return handle.read()
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue

    raise ValueError(f"Could not decode {filepath} with any known encoding") from last_error


def extract_content(filepath: str) -> dict[str, Any]:
    """
    Extract filename, filepath, extension, and raw text content from a
    single file on disk.

    Parameters
    ----------
    filepath:
        Absolute or relative path to the file to read.

    Returns
    -------
    A dictionary with keys `filename`, `filepath`, `extension`, and
    `content`.

    Raises
    ------
    FileNotFoundError:
        If `filepath` does not point to an existing file.
    ValueError:
        If the file's content could not be decoded as text.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")

    filename = os.path.basename(filepath)
    extension = os.path.splitext(filename)[1].lower()

    content = _read_file_text(filepath)

    logger.debug("Extracted %d characters from %s", len(content), filepath)

    return {
        "filename": filename,
        "filepath": os.path.abspath(filepath),
        "extension": extension,
        "content": content,
    }