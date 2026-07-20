"""
code_chunker.py

AST-based chunker for Python source files. Produces:
  - one combined "imports" chunk for all top-level import statements,
  - one chunk per top-level function (sync or async),
  - one chunk per top-level class, with all of its methods included,
  - one combined "__main__" chunk for any other top-level executable
    code.

Falls back to paragraph-based text chunking (via `text_chunker.py`) if
the source cannot be parsed as Python, or if parsing succeeds but no
top-level constructs are found.
"""

from __future__ import annotations

import ast
from typing import Any

from logger import logger
from text_chunker import chunk_plain_text

_IMPORT_NODE_TYPES = (ast.Import, ast.ImportFrom)
_DEFINITION_NODE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _get_segment(source_lines: list[str], node: ast.AST) -> str:
    """
    Extract the exact source text spanned by `node`, including any
    decorators attached to it.
    """
    start_line = node.lineno
    decorator_list = getattr(node, "decorator_list", [])
    if decorator_list:
        start_line = min(start_line, min(d.lineno for d in decorator_list))

    end_line = getattr(node, "end_lineno", node.lineno)

    segment_lines = source_lines[start_line - 1:end_line]
    return "\n".join(segment_lines).rstrip("\n")


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


def _chunk_ast(
    file_data: dict[str, Any],
    tree: ast.Module,
    source_lines: list[str],
) -> list[dict[str, Any]]:
    """
    Walk the top-level statements of a parsed module and group them into
    ordered chunk contents: imports first, then each function/class in
    source order, then a trailing "__main__" chunk for everything else.
    """
    filename = file_data.get("filename", "")
    filepath = file_data.get("filepath", "")
    extension = file_data.get("extension", "")

    import_segments: list[str] = []
    definition_segments: list[str] = []
    other_segments: list[str] = []

    for node in tree.body:
        if isinstance(node, _IMPORT_NODE_TYPES):
            import_segments.append(_get_segment(source_lines, node))
        elif isinstance(node, _DEFINITION_NODE_TYPES):
            definition_segments.append(_get_segment(source_lines, node))
        else:
            other_segments.append(_get_segment(source_lines, node))

    ordered_contents: list[str] = []

    if import_segments:
        ordered_contents.append("# imports\n" + "\n".join(import_segments))

    ordered_contents.extend(definition_segments)

    if other_segments:
        ordered_contents.append("# __main__\n" + "\n\n".join(other_segments))

    return [
        _build_chunk(filename, filepath, extension, chunk_id, content_text)
        for chunk_id, content_text in enumerate(ordered_contents, start=1)
    ]


def chunk_code(file_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Chunk a Python source file into function/class/import/"__main__"
    chunks using the `ast` module.

    Parameters
    ----------
    file_data:
        Dictionary produced by `extract_content`, containing at least
        `filename`, `filepath`, `extension`, and `content`.

    Returns
    -------
    A list of chunk dictionaries preserving the original metadata and
    adding a sequential `chunk_id` starting at 1. Falls back to
    `chunk_plain_text` if the source is not valid Python, or if no
    top-level constructs are found after a successful parse.
    """
    filename = file_data.get("filename", "")
    filepath = file_data.get("filepath", "")
    content = file_data.get("content", "") or ""

    if not content.strip():
        logger.warning("Empty code content for %s; skipping chunking.", filepath)
        return []

    try:
        tree = ast.parse(content, filename=filepath or filename or "<unknown>")
    except SyntaxError as exc:
        logger.warning(
            "Failed to parse %s as Python (%s); falling back to text chunking.",
            filepath,
            exc,
        )
        return chunk_plain_text(file_data)

    source_lines = content.splitlines()
    chunks = _chunk_ast(file_data, tree, source_lines)

    if not chunks:
        logger.warning(
            "No top-level constructs found in %s; falling back to text chunking.",
            filepath,
        )
        return chunk_plain_text(file_data)

    logger.debug("Chunked %s into %d chunk(s) [code].", filepath, len(chunks))
    return chunks