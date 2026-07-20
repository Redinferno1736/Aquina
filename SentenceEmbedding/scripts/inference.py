"""
inference.py

Standalone inference bridge between the Aquina Sentence Embedding model
and an external process (e.g. a Rust/Tauri backend). The model and
index are loaded exactly once at startup; after that, this script
reads one JSON request per line from stdin and writes one JSON
response per line to stdout, so the calling process can spawn this as
a long-lived sidecar instead of paying the model-load cost on every
query.

Protocol
--------
stdin  (one JSON object per line):
    {"query": "open the resume tracker", "top_k": 5, "min_score": 0.20}
    {"command": "ping"}
    {"command": "exit"}

stdout (one JSON object per line, always exactly one response per
        request line, in order):
    {"ok": true, "results": [ {...}, {...} ]}
    {"ok": true, "pong": true}
    {"ok": false, "error": "message describing what went wrong"}

IMPORTANT: stdout carries ONLY JSON, one object per line. All logging
goes to stderr via logger.py, so the calling process (Rust/Tauri) can
safely parse stdout line-by-line without log noise mixed in. This
script never calls print() for anything other than a JSON response
line.

Standalone testing
-------------------
This script can also be run in single-shot CLI mode for manual testing
without wiring up stdin piping:

    python inference.py --query "open the resume tracker" --top-k 5

Rust/Tauri integration sketch
------------------------------
Spawn this script once as a child process (e.g. via
`std::process::Command` / `tauri::api::process::Command`), keep its
stdin/stdout pipes open, write one JSON line per query, and read one
JSON line back. Do not spawn a new process per query — the whole point
of this protocol is to load the model once and keep it warm.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from config import TOP_K
from logger import logger
from search_engine import SearchEngine


class InferenceService:
    """
    Wraps a SearchEngine instance and exposes a JSON-in/JSON-out
    request handler suitable for driving from an external process.
    """

    def __init__(self) -> None:
        logger.info("InferenceService starting up...")
        self._engine = SearchEngine()
        logger.info("InferenceService ready.")

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Handle a single decoded JSON request and return a JSON-safe
        response dictionary. Never raises — all failures are captured
        and returned as {"ok": false, "error": "..."}.
        """
        command = request.get("command")

        if command == "ping":
            return {"ok": True, "pong": True}

        if command == "exit":
            return {"ok": True, "bye": True}

        query = request.get("query")
        if not isinstance(query, str) or not query.strip():
            return {"ok": False, "error": "Request must include a non-empty 'query' string."}

        top_k = request.get("top_k", TOP_K)
        min_score = request.get("min_score", 0.20)

        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"'top_k' must be an integer, got {top_k!r}."}

        try:
            min_score = float(min_score)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"'min_score' must be a number, got {min_score!r}."}

        try:
            results = self._engine.search(query=query, top_k=top_k, min_score=min_score)
        except (RuntimeError, ValueError) as exc:
            logger.error("Search failed for query %r: %s", query, exc)
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("Unexpected error handling query %r", query)
            return {"ok": False, "error": f"Unexpected error: {exc}"}

        return {"ok": True, "results": results}

    def run_stdin_loop(self) -> None:
        """
        Continuously read JSON request lines from stdin and write JSON
        response lines to stdout until stdin closes or an 'exit'
        command is received.
        """
        logger.info("Entering stdin/stdout request loop.")

        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                self._write_response({"ok": False, "error": f"Invalid JSON: {exc}"})
                continue

            if not isinstance(request, dict):
                self._write_response(
                    {"ok": False, "error": "Request must be a JSON object."}
                )
                continue

            response = self.handle_request(request)
            self._write_response(response)

            if request.get("command") == "exit":
                logger.info("Exit command received; shutting down inference loop.")
                break

        logger.info("Stdin closed; inference loop ended.")

    @staticmethod
    def _write_response(response: dict[str, Any]) -> None:
        """Write exactly one JSON response as a single stdout line and flush immediately."""
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def _run_single_shot(query: str, top_k: int, min_score: float) -> None:
    """Run one query through the engine and print the JSON result, for manual CLI testing."""
    service = InferenceService()
    response = service.handle_request(
        {"query": query, "top_k": top_k, "min_score": min_score}
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aquina Sentence Embedding inference bridge."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single query in one-shot mode and print the JSON result, "
        "instead of entering the stdin/stdout loop.",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of results to return.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.20,
        help="Minimum similarity score for a result to be included.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.query is not None:
        _run_single_shot(query=args.query, top_k=args.top_k, min_score=args.min_score)
        return

    service = InferenceService()
    service.run_stdin_loop()


if __name__ == "__main__":
    main()