"""
scripts/build_dataset.py

Merges the two raw datasets produced by scripts/download_leetcode.py
(data/leetcode_dataset.json) and scripts/scrape_codeforces.py
(data/codeforces_dataset.json) into one standardized dataset at
data/unified_dataset.json.

Important, source-specific caveat (read before assuming both sources
produce equally rich records):

  The Codeforces official API (problemset.problems) never returns full
  problem-statement text -- only metadata (name, tags, rating, contest
  info). Getting the actual statement would require scraping the HTML
  problem page per-problem, which is exactly what this project avoids
  for rate-limit/reliability reasons (see scripts/scrape_codeforces.py).
  So for Codeforces, `content` is synthesized from the metadata that IS
  available (title + tags + rating) rather than left as the real
  problem statement. This is clearly marked per-record via the
  `content_is_synthetic` field so nothing downstream (including a
  future Bedrock tech-debt-report step) silently treats it as full
  statement text. Pass --strict-content to instead require real
  content and drop every Codeforces record (see --help).

Unified schema (see the `UnifiedProblem` dataclass below):
    id                  : str
    source               : "leetcode" | "codeforces"
    title                : str
    difficulty_level     : str | int | None
    tags                 : list[str]
    content               : str | None
    content_is_synthetic : bool

Usage:
    python -m scripts.build_dataset
    python -m scripts.build_dataset --strict-content
    python -m scripts.build_dataset --leetcode data/leetcode_dataset.json --codeforces data/codeforces_dataset.json --output data/unified_dataset.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_dataset")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_LEETCODE_PATH = os.path.join(_PROJECT_ROOT, "data", "leetcode_dataset.json")
_DEFAULT_CODEFORCES_PATH = os.path.join(_PROJECT_ROOT, "data", "codeforces_dataset.json")
_DEFAULT_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "unified_dataset.json")


@dataclass
class UnifiedProblem:
    """The single standardized shape every problem record is normalized into."""

    id: str
    source: str  # "leetcode" | "codeforces"
    title: str
    difficulty_level: str | int | None = None
    tags: list[str] = field(default_factory=list)
    content: str | None = None
    content_is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BuildStats:
    """Counters reported at the end of a build run."""

    leetcode_loaded: int = 0
    leetcode_dropped: int = 0
    codeforces_loaded: int = 0
    codeforces_dropped: int = 0
    codeforces_synthetic_content: int = 0
    total_saved: int = 0

    def log_summary(self) -> None:
        logger.info(
            "LeetCode:    loaded=%d  dropped=%d  kept=%d",
            self.leetcode_loaded, self.leetcode_dropped,
            self.leetcode_loaded - self.leetcode_dropped,
        )
        logger.info(
            "Codeforces:  loaded=%d  dropped=%d  kept=%d  (synthetic content: %d)",
            self.codeforces_loaded, self.codeforces_dropped,
            self.codeforces_loaded - self.codeforces_dropped,
            self.codeforces_synthetic_content,
        )
        logger.info("Total records written: %d", self.total_saved)


class DatasetBuildError(Exception):
    """Raised when a raw dataset file cannot be loaded or has an unusable shape."""


def load_raw_dataset(path: str, source_name: str) -> list[dict[str, Any]]:
    """
    Load a raw dataset JSON file. Returns an empty list (with a logged
    warning, not a raised error) if the file doesn't exist, so a
    missing/not-yet-downloaded source doesn't block building a unified
    dataset from whatever sources ARE available.
    """
    if not os.path.exists(path):
        logger.warning("%s dataset not found at %s; skipping this source.", source_name, path)
        return []

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetBuildError(f"Failed to load {source_name} dataset from {path}: {exc}") from exc

    if not isinstance(data, list):
        raise DatasetBuildError(
            f"{source_name} dataset at {path} is not a JSON list (got {type(data).__name__})"
        )

    logger.info("Loaded %d raw %s records from %s", len(data), source_name, path)
    return data


def _clean_tags(raw_tags: Any) -> list[str]:
    """Normalize a tags field into a list[str], defaulting to [] for anything unusable."""
    if not raw_tags:
        return []
    if not isinstance(raw_tags, list):
        return []
    return [str(tag).strip() for tag in raw_tags if tag is not None and str(tag).strip()]


def _clean_text(value: Any) -> str | None:
    """Normalize a string-ish field: strip whitespace, treat empty string as None."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_leetcode_record(raw: dict[str, Any]) -> UnifiedProblem | None:
    """
    Map one raw LeetCode record (from scripts/download_leetcode.py's
    output shape) onto UnifiedProblem. Returns None if the record lacks
    a usable title or id -- callers are expected to still run the
    content-completeness check separately (see build_unified_dataset).
    """
    problem_id = raw.get("problem_id") or raw.get("title_slug")
    title = _clean_text(raw.get("title"))

    if not problem_id or not title:
        return None

    return UnifiedProblem(
        id=str(problem_id),
        source="leetcode",
        title=title,
        difficulty_level=raw.get("difficulty"),
        tags=_clean_tags(raw.get("topic_tags")),
        content=_clean_text(raw.get("description_html")),
        content_is_synthetic=False,
    )


def normalize_codeforces_record(raw: dict[str, Any]) -> UnifiedProblem | None:
    """
    Map one raw Codeforces record (from scripts/scrape_codeforces.py's
    output shape) onto UnifiedProblem. Codeforces' official API never
    returns problem-statement text, so `content` is synthesized from
    available metadata and flagged via content_is_synthetic=True (see
    module docstring for why).
    """
    problem_id = raw.get("problem_id")
    title = _clean_text(raw.get("name"))

    if not problem_id or not title:
        return None

    tags = _clean_tags(raw.get("tags"))
    rating = raw.get("rating")

    synthetic_parts = [title]
    if rating is not None:
        synthetic_parts.append(f"Difficulty rating: {rating}.")
    if tags:
        synthetic_parts.append(f"Tags: {', '.join(tags)}.")
    synthetic_content = " ".join(synthetic_parts)

    return UnifiedProblem(
        id=str(problem_id),
        source="codeforces",
        title=title,
        difficulty_level=rating,
        tags=tags,
        content=synthetic_content,
        content_is_synthetic=True,
    )


def build_unified_dataset(
    leetcode_records: list[dict[str, Any]],
    codeforces_records: list[dict[str, Any]],
    stats: BuildStats,
    strict_content: bool = False,
) -> list[UnifiedProblem]:
    """
    Normalize both raw record lists into UnifiedProblem, dropping any
    record that lacks a title or (per --strict-content) real,
    non-synthetic content. Updates `stats` in place.
    """
    stats.leetcode_loaded = len(leetcode_records)
    stats.codeforces_loaded = len(codeforces_records)

    unified: list[UnifiedProblem] = []

    for raw in leetcode_records:
        problem = normalize_leetcode_record(raw)
        if problem is None:
            stats.leetcode_dropped += 1
            continue
        if problem.content is None:
            # No title/id issue, but description was never fetched
            # (e.g. --skip-details was used, or the per-problem detail
            # fetch failed) or was genuinely empty -- not enough
            # actionable data to analyze.
            stats.leetcode_dropped += 1
            continue
        unified.append(problem)

    for raw in codeforces_records:
        problem = normalize_codeforces_record(raw)
        if problem is None:
            stats.codeforces_dropped += 1
            continue
        if problem.content_is_synthetic:
            if strict_content:
                stats.codeforces_dropped += 1
                continue
            stats.codeforces_synthetic_content += 1
        unified.append(problem)

    return unified


def save_unified_dataset(problems: list[UnifiedProblem], output_path: str) -> None:
    """Write the unified problem list to `output_path` as pretty-printed JSON."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    records = [p.to_dict() for p in problems]
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise DatasetBuildError(f"Failed to write output file '{output_path}': {exc}") from exc

    logger.info("Wrote %d unified records to %s", len(records), output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge raw LeetCode + Codeforces datasets into one standardized dataset."
    )
    parser.add_argument("--leetcode", default=_DEFAULT_LEETCODE_PATH, help=f"Path to raw LeetCode dataset (default: {_DEFAULT_LEETCODE_PATH})")
    parser.add_argument("--codeforces", default=_DEFAULT_CODEFORCES_PATH, help=f"Path to raw Codeforces dataset (default: {_DEFAULT_CODEFORCES_PATH})")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT_PATH, help=f"Output path for the unified dataset (default: {_DEFAULT_OUTPUT_PATH})")
    parser.add_argument(
        "--strict-content", action="store_true",
        help=(
            "Drop Codeforces records instead of giving them synthesized "
            "content, since the Codeforces API provides no real problem "
            "statement text. Off by default (synthesized content is kept) "
            "since dropping would discard the entire Codeforces dataset."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = BuildStats()

    try:
        leetcode_records = load_raw_dataset(args.leetcode, "LeetCode")
        codeforces_records = load_raw_dataset(args.codeforces, "Codeforces")

        if not leetcode_records and not codeforces_records:
            logger.error("Neither dataset produced any records; aborting without writing output.")
            return 1

        unified = build_unified_dataset(
            leetcode_records, codeforces_records, stats, strict_content=args.strict_content
        )

        if not unified:
            logger.error("No records survived normalization/cleaning; aborting without writing output.")
            return 1

        stats.total_saved = len(unified)
        save_unified_dataset(unified, args.output)
    except DatasetBuildError as exc:
        logger.error("Failed to build unified dataset: %s", exc)
        return 1

    stats.log_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())