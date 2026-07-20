"""
markdown_chunker.py

Heading-based chunker for Markdown files. Each heading (#, ##, ###, ####)
starts a new chunk containing that heading plus the content beneath it,
up to the next heading of any level. Sections larger than
MARKDOWN_MAX_SECTION_WORDS words are further split into overlapping
sub-chunks, each still prefixed with the original heading text. Lines
inside fenced code blocks (``` or ~~~) are never treated as headings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import CHUNK_OVERLAP_WORDS, MARKDOWN_MAX_SECTION_WORDS
from logger import logger

FENCE_MARKERS = ("```", "~~~")
MIN_HEADING_LEVEL = 1
MAX_HEADING_LEVEL = 4


@dataclass
class _Section:
    heading: str | None
    level: int
    lines: list[str] = field(default_factory=list)


def _is_heading(line: str) -> tuple[bool, int]:
    """Determine whether `line` is a Markdown heading (levels 1-4)."""
    stripped = line.strip()
    if not stripped.startswith("#"):
        return False, 0

    hash_count = 0
    for char in stripped:
        if char == "#":
            hash_count += 1
        else:
            break

    if hash_count < MIN_HEADING_LEVEL or hash_count > MAX_HEADING_LEVEL:
        return False, 0

    remainder = stripped[hash_count:]
    if not remainder.startswith(" "):
        return False, 0

    return True, hash_count


def _split_into_sections(content: str) -> list[_Section]:
    """Split Markdown content into sections, one per heading."""
    lines = content.splitlines()
    sections: list[_Section] = []
    current = _Section(heading=None, level=0)
    in_fence = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(FENCE_MARKERS):
            in_fence = not in_fence
            current.lines.append(line)
            continue

        if not in_fence:
            is_heading, level = _is_heading(line)
            if is_heading:
                if current.heading is not None or current.lines:
                    sections.append(current)
                current = _Section(heading=stripped, level=level)
                continue

        current.lines.append(line)

    if current.heading is not None or current.lines:
        sections.append(current)

    return sections


def _words(text: str) -> list[str]:
    return text.split()


def _chunk_words_with_overlap(
    words: list[str],
    target_words: int,
    overlap_words: int,
) -> list[list[str]]:
    """Split a flat list of words into overlapping windows of ~target_words words."""
    if len(words) <= target_words:
        return [words] if words else []

    chunks: list[list[str]] = []
    start = 0
    while start < len(words):
        end = start + target_words
        chunks.append(words[start:end])
        if end >= len(words):
            break
        start = end - overlap_words
    return chunks


def _section_to_chunk_contents(section: _Section) -> list[str]:
    """Render a section into one or more chunk content strings, heading included."""
    heading_text = section.heading if section.heading else ""
    body_text = "\n".join(section.lines).strip()

    if not heading_text and not body_text:
        return []

    full_text = f"{heading_text}\n{body_text}".strip() if heading_text else body_text
    word_count = len(_words(full_text))

    if word_count <= MARKDOWN_MAX_SECTION_WORDS:
        return [full_text]

    body_words = _words(body_text)
    sub_word_chunks = _chunk_words_with_overlap(
        body_words, MARKDOWN_MAX_SECTION_WORDS, CHUNK_OVERLAP_WORDS
    )

    contents: list[str] = []
    for sub_words in sub_word_chunks:
        sub_body = " ".join(sub_words)
        contents.append(f"{heading_text}\n{sub_body}" if heading_text else sub_body)
    return contents


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


def chunk_markdown(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Chunk a Markdown file by heading, splitting oversized sections while
    preserving the heading in every resulting chunk.

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
        logger.warning("Empty markdown content for %s; skipping chunking.", filepath)
        return []

    sections = _split_into_sections(content)
    if not sections:
        logger.warning("No sections found in %s; skipping chunking.", filepath)
        return []

    chunk_contents: list[str] = []
    for section in sections:
        chunk_contents.extend(_section_to_chunk_contents(section))

    chunks = [
        _build_chunk(filename, filepath, extension, chunk_id, content_text)
        for chunk_id, content_text in enumerate(chunk_contents, start=1)
    ]

    logger.debug("Chunked %s into %d chunk(s) [markdown].", filepath, len(chunks))
    return chunks