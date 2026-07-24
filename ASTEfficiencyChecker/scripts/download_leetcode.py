"""
scripts/download_leetcode.py

Fetches problem data for every public (non-paid-only) problem on
LeetCode via LeetCode's public GraphQL API and writes it to
<project_root>/data/leetcode_dataset.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import requests

try:
    import config
    _DEFAULT_OUTPUT_PATH = os.path.join(config.DATA_DIR, "leetcode_dataset.json")
except ImportError:
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _DEFAULT_OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "data", "leetcode_dataset.json")

# --- FIXED LOGGING SETUP ---
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("scripts.download_leetcode")
# ---------------------------

_GRAPHQL_URL = "https://leetcode.com/graphql"

_HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://leetcode.com/problemset/all/",
    "User-Agent": (
        "Mozilla/5.0 (compatible; ASTEfficiencyChecker-DatasetBuilder/1.0; "
        "+https://github.com/) research/education dataset collection"
    ),
}

_PROBLEM_LIST_QUERY = """
query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
  problemsetQuestionList: questionList(
    categorySlug: $categorySlug
    limit: $limit
    skip: $skip
    filters: $filters
  ) {
    total: totalNum
    questions: data {
      questionFrontendId
      title
      titleSlug
      difficulty
      isPaidOnly
      topicTags {
        name
        slug
      }
    }
  }
}
"""

_PROBLEM_DETAIL_QUERY = """
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionFrontendId
    title
    titleSlug
    difficulty
    content
    isPaidOnly
    likes
    dislikes
    hints
    exampleTestcases
    topicTags {
      name
      slug
    }
    codeSnippets {
      lang
      langSlug
      code
    }
    stats
  }
}
"""

_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 2.0
_RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}


@dataclass
class DownloadStats:
    total_available: int = 0
    metadata_fetched: int = 0
    details_fetched: int = 0
    details_skipped_paid: int = 0
    details_failed: int = 0
    already_done_on_resume: int = 0


class LeetCodeGraphQLError(Exception):
    pass


class LeetCodeDownloader:
    def __init__(
        self,
        output_path: str = _DEFAULT_OUTPUT_PATH,
        delay_seconds: float = 1.5,
        jitter_seconds: float = 0.5,
        skip_details: bool = False,
        limit: int | None = None,
    ) -> None:
        self.output_path = output_path
        self.delay_seconds = delay_seconds
        self.jitter_seconds = jitter_seconds
        self.skip_details = skip_details
        self.limit = limit

        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.stats = DownloadStats()
        self._problems: dict[str, dict[str, Any]] = {}

    def run(self) -> None:
        self._ensure_output_dir()
        self._load_existing_output()

        logger.info("Fetching problem list from LeetCode GraphQL API...")
        problem_list = self._fetch_full_problem_list()
        self.stats.total_available = len(problem_list)
        logger.info("Discovered %d problems (limit=%s).", len(problem_list), self.limit)

        for index, summary in enumerate(problem_list, start=1):
            slug = summary["titleSlug"]

            if self._already_has_full_record(slug):
                self.stats.already_done_on_resume += 1
                continue

            self._merge_summary(slug, summary)
            self.stats.metadata_fetched += 1

            if not self.skip_details and not summary.get("isPaidOnly", False):
                self._fetch_and_merge_detail(slug)
            elif summary.get("isPaidOnly", False):
                self.stats.details_skipped_paid += 1
                logger.debug("Skipping detail fetch for paid-only problem: %s", slug)

            self._save_output()

            if index % 25 == 0 or index == len(problem_list):
                logger.info(
                    "Progress: %d/%d problems processed (%d resumed, %d detail failures).",
                    index, len(problem_list),
                    self.stats.already_done_on_resume, self.stats.details_failed,
                )

        self._save_output()
        self._log_final_summary()

    def _fetch_full_problem_list(self) -> list[dict[str, Any]]:
        page_size = 100
        skip = 0
        all_questions: list[dict[str, Any]] = []
        total_num: int | None = None

        while True:
            variables = {
                "categorySlug": "",
                "limit": page_size,
                "skip": skip,
                "filters": {},
            }
            payload = self._post_graphql(_PROBLEM_LIST_QUERY, variables)

            try:
                result = payload["data"]["problemsetQuestionList"]
                questions = result["questions"]
                total_num = result["total"]
            except (KeyError, TypeError) as exc:
                raise LeetCodeGraphQLError(f"Unexpected problem-list response shape: {payload}") from exc

            if not questions:
                break

            all_questions.extend(questions)
            skip += len(questions)

            if self.limit is not None and len(all_questions) >= self.limit:
                all_questions = all_questions[: self.limit]
                break

            if total_num is not None and skip >= total_num:
                break

            self._rate_limit_pause()

        return all_questions

    def _fetch_and_merge_detail(self, slug: str) -> None:
        self._rate_limit_pause()
        try:
            payload = self._post_graphql(_PROBLEM_DETAIL_QUERY, {"titleSlug": slug})
            question = payload["data"]["question"]
            if question is None:
                raise LeetCodeGraphQLError(f"No question data returned for slug '{slug}'")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch details for '%s': %s", slug, exc)
            self.stats.details_failed += 1
            self._problems[slug]["detail_fetch_error"] = str(exc)
            return

        record = self._problems[slug]
        record["description_html"] = question.get("content")
        record["hints"] = question.get("hints", [])
        record["example_testcases"] = question.get("exampleTestcases")
        record["likes"] = question.get("likes")
        record["dislikes"] = question.get("dislikes")
        record["code_snippets"] = {
            snippet["langSlug"]: snippet["code"]
            for snippet in question.get("codeSnippets", []) or []
        }
        record["stats_raw"] = question.get("stats")
        record["detail_fetched"] = True
        self.stats.details_fetched += 1

    def _merge_summary(self, slug: str, summary: dict[str, Any]) -> None:
        self._problems[slug] = {
            "problem_id": f"leetcode_{summary['questionFrontendId']}",
            "question_frontend_id": summary["questionFrontendId"],
            "title": summary["title"],
            "title_slug": slug,
            "difficulty": summary["difficulty"],
            "is_paid_only": summary.get("isPaidOnly", False),
            "topic_tags": [tag["name"] for tag in summary.get("topicTags", []) or []],
            "detail_fetched": False,
        }

    def _already_has_full_record(self, slug: str) -> bool:
        record = self._problems.get(slug)
        if record is None:
            return False
        if self.skip_details:
            return True
        if record.get("is_paid_only"):
            return True
        return bool(record.get("detail_fetched"))

    def _post_graphql(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        last_exception: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.session.post(
                    _GRAPHQL_URL,
                    json={"query": query, "variables": variables},
                    timeout=30,
                )

                if response.status_code in _RETRYABLE_STATUS_CODES:
                    wait_seconds = self._backoff_delay(attempt)
                    logger.warning(
                        "HTTP %d from LeetCode (attempt %d/%d); backing off %.1fs.",
                        response.status_code, attempt, _MAX_RETRIES, wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                response.raise_for_status()
                payload = response.json()

                if "errors" in payload and payload["errors"]:
                    raise LeetCodeGraphQLError(f"GraphQL error(s): {payload['errors']}")

                return payload

            except (requests.RequestException, LeetCodeGraphQLError, ValueError) as exc:
                last_exception = exc
                wait_seconds = self._backoff_delay(attempt)
                logger.warning(
                    "Request failed (attempt %d/%d): %s -- retrying in %.1fs.",
                    attempt, _MAX_RETRIES, exc, wait_seconds,
                )
                time.sleep(wait_seconds)

        raise LeetCodeGraphQLError(
            f"Exhausted {_MAX_RETRIES} retries for GraphQL request; last error: {last_exception}"
        )

    def _backoff_delay(self, attempt: int) -> float:
        return (_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))) + random.uniform(0, 1.0)

    def _rate_limit_pause(self) -> None:
        time.sleep(self.delay_seconds + random.uniform(0, self.jitter_seconds))

    def _ensure_output_dir(self) -> None:
        directory = os.path.dirname(self.output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _load_existing_output(self) -> None:
        if not os.path.exists(self.output_path):
            logger.info("No existing output at %s; starting a fresh download.", self.output_path)
            return

        try:
            with open(self.output_path, "r", encoding="utf-8") as fh:
                existing_records = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read existing output at %s (%s); starting fresh instead of resuming.",
                self.output_path, exc,
            )
            return

        for record in existing_records:
            slug = record.get("title_slug")
            if slug:
                self._problems[slug] = record

        logger.info(
            "Loaded %d previously downloaded problems from %s; resuming.",
            len(self._problems), self.output_path,
        )

    def _save_output(self) -> None:
        records = list(self._problems.values())
        records.sort(key=lambda r: int(r["question_frontend_id"]))

        temp_path = f"{self.output_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.output_path)

    def _log_final_summary(self) -> None:
        logger.info(
            "Download complete. total_available=%d metadata_fetched=%d "
            "details_fetched=%d details_skipped_paid=%d details_failed=%d "
            "already_done_on_resume=%d",
            self.stats.total_available,
            self.stats.metadata_fetched,
            self.stats.details_fetched,
            self.stats.details_skipped_paid,
            self.stats.details_failed,
            self.stats.already_done_on_resume,
        )
        logger.info("Output written to %s", self.output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LeetCode problem data via the public GraphQL API."
    )
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {_DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Base delay in seconds between requests (default: 1.5)",
    )
    parser.add_argument(
        "--jitter", type=float, default=0.5,
        help="Extra random jitter (0..N seconds) added to each delay (default: 0.5)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only fetch the first N problems from the list (useful for a smoke test)",
    )
    parser.add_argument(
        "--skip-details", action="store_true",
        help="Only fetch problem-list metadata (title/difficulty/tags), skip full per-problem detail fetches",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    downloader = LeetCodeDownloader(
        output_path=args.output,
        delay_seconds=args.delay,
        jitter_seconds=args.jitter,
        skip_details=args.skip_details,
        limit=args.limit,
    )

    try:
        downloader.run()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; progress up to this point has already been saved.")
        sys.exit(1)


if __name__ == "__main__":
    main()