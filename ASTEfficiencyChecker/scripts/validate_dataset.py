"""
scripts/scrape_codeforces.py

Fetches problem metadata for every problem in the Codeforces problemset
via the official Codeforces API (problemset.problems endpoint) and
writes it to <project_root>/data/codeforces_dataset.json.

No HTML scraping is involved: problemset.problems returns the entire
problem catalog (several thousand problems) in a single bulk response,
so this script runs in seconds rather than hours and never needs
pagination or per-problem requests.

Self-contained: only uses the standard library plus `requests`, and
logs via the standard `logging` module (no project-specific utils),
so it can be copy-pasted and run on its own if needed.

Usage:
    python -m scripts.scrape_codeforces
    python -m scripts.scrape_codeforces --output data/custom.json
    python -m scripts.scrape_codeforces --timeout 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scrape_codeforces")

_API_URL = "https://codeforces.com/api/problemset.problems"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "codeforces_dataset.json")


class CodeforcesAPIError(Exception):
    """Raised when the Codeforces API returns a non-OK status or an unexpected response shape."""


def fetch_problemset(timeout: int = 30) -> dict[str, Any]:
    """
    Call the problemset.problems endpoint and return the raw parsed
    JSON payload. Raises CodeforcesAPIError on any network failure,
    non-200 HTTP status, non-JSON body, or an API-level status that
    isn't "OK".
    """
    logger.info("Requesting %s ...", _API_URL)
    try:
        response = requests.get(_API_URL, timeout=timeout)
    except requests.RequestException as exc:
        raise CodeforcesAPIError(f"Network error while calling Codeforces API: {exc}") from exc

    if response.status_code != 200:
        raise CodeforcesAPIError(
            f"Codeforces API returned HTTP {response.status_code}: {response.text[:500]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise CodeforcesAPIError(f"Codeforces API did not return valid JSON: {exc}") from exc

    if payload.get("status") != "OK":
        comment = payload.get("comment", "no comment provided")
        raise CodeforcesAPIError(f"Codeforces API status was not OK: {comment}")

    return payload


def extract_problems(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract and normalize the fields we care about (problem id, name,
    tags, rating/difficulty) from the raw API payload's `result.problems`
    list.

    The API also returns a parallel `result.problemStatistics` list
    (solved-count per problem); it's merged in here by (contestId, index)
    since it's a genuinely useful signal (a low solve-count problem with
    a low rating can indicate an unusual/tricky problem) and costs
    nothing extra since it comes back in the same response.
    """
    try:
        result = payload["result"]
        raw_problems = result["problems"]
        raw_statistics = result.get("problemStatistics", [])
    except (KeyError, TypeError) as exc:
        raise CodeforcesAPIError(f"Unexpected payload shape, missing key: {exc}") from exc

    solved_count_by_key: dict[tuple[int, str], int] = {}
    for stat in raw_statistics:
        contest_id = stat.get("contestId")
        index = stat.get("index")
        if contest_id is not None and index is not None:
            solved_count_by_key[(contest_id, index)] = stat.get("solvedCount", 0)

    problems: list[dict[str, Any]] = []
    skipped = 0

    for raw in raw_problems:
        contest_id = raw.get("contestId")
        index = raw.get("index")
        name = raw.get("name")

        if contest_id is None or index is None or not name:
            # A small number of problems (e.g. some gym/April-fools
            # problems) can be missing fields needed to form a stable
            # id; skip rather than emit a malformed record.
            skipped += 1
            continue

        problem_id = f"codeforces_{contest_id}{index}"
        solved_count = solved_count_by_key.get((contest_id, index))

        problems.append({
            "problem_id": problem_id,
            "contest_id": contest_id,
            "index": index,
            "name": name,
            "rating": raw.get("rating"),
            "tags": raw.get("tags", []),
            "problem_type": raw.get("type"),
            "points": raw.get("points"),
            "solved_count": solved_count,
        })

    if skipped:
        logger.warning("Skipped %d problem(s) missing required fields (contestId/index/name).", skipped)

    return problems


def save_dataset(problems: list[dict[str, Any]], output_path: str) -> None:
    """
    Write `problems` to `output_path` as pretty-printed JSON, creating
    the parent directory (e.g. data/) first if it does not exist yet.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(problems, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise CodeforcesAPIError(f"Failed to write output file '{output_path}': {exc}") from exc

    logger.info("Wrote %d problems to %s", len(problems), output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the full Codeforces problemset via the official API."
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {_DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--timeout", type=int, default=30,
        help="HTTP request timeout in seconds (default: 30)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        payload = fetch_problemset(timeout=args.timeout)
        problems = extract_problems(payload)

        if not problems:
            logger.error("No problems extracted from the API response; aborting without writing output.")
            return 1

        save_dataset(problems, args.output)
    except CodeforcesAPIError as exc:
        logger.error("Failed to build Codeforces dataset: %s", exc)
        return 1

    logger.info("Done. %d problems available in %s", len(problems), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())