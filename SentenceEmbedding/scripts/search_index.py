"""
search_index.py

Thin CLI wrapper around SearchEngine. Contains no search logic itself —
it constructs a SearchEngine once, then repeatedly prompts the user for
a query, calls search(), and prints the formatted results.
"""

from __future__ import annotations

from typing import Any

from logger import logger
from search_engine import SearchEngine


def _print_results(results: list[dict[str, Any]]) -> None:
    """Print a list of search results in a readable format."""
    if not results:
        print("No results found.")
        return

    for result in results:
        print(f"\nRank:      {result['rank']}")
        print(f"Score:     {result['score']:.4f}")
        print(f"Filename:  {result['filename']}")
        print(f"Filepath:  {result['filepath']}")
        print(f"Chunk ID:  {result['chunk_id']}")
        print(f"Preview:   {result['preview']}")


def main() -> None:
    """Run the interactive search CLI."""
    try:
        engine = SearchEngine()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("Could not start SearchEngine: %s", exc)
        return

    print("Aquina Semantic Search — type your query, or 'exit' to quit.")
    while True:
        query = input("\nSearch> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        try:
            results = engine.search(query)
        except (RuntimeError, ValueError) as exc:
            logger.error("Search failed: %s", exc)
            continue

        _print_results(results)


if __name__ == "__main__":
    main()