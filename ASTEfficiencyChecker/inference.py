"""
ast_inference_engine.py

Production inference bridge for the AST-based efficiency checker (Aquina B2).

Loads a precomputed dataset of known-optimal algorithm embeddings, parses an
incoming source-code snippet (Python / Java / C) into an AST, generates an
embedding for it, and matches it against the known-optimal set to estimate
time/space complexity and pattern tags.

Designed to be invoked by a Rust backend via `std::process::Command`:
    echo '{"code": "...", "language": "python"}' | python ast_inference_engine.py

Only a single JSON object is ever written to stdout. All logging, warnings,
and errors go to stderr so the Rust side can deserialize stdout cleanly with
serde_json without any extra parsing/cleanup step.

Requirements:
    pip install tree-sitter tree-sitter-languages numpy
"""

from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# tree-sitter-languages ships prebuilt grammars for python/java/c (and many more)
# under one consistent API, so we don't need to juggle pycparser/javalang/ast
# separately for each language.
try:
    from tree_sitter_languages import get_parser
except ImportError:  # pragma: no cover
    get_parser = None  # handled at runtime with a clear error


# --------------------------------------------------------------------------- #
# Logging: stderr only. Never let logging touch stdout.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="[ast_inference_engine] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = {"python", "java", "c"}


@dataclass
class MatchResult:
    """Structured result of matching a snippet against the known-optimal set."""

    time_complexity: str
    space_complexity: str
    pattern_tags: list[str] = field(default_factory=list)
    matched_problem_id: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "pattern_tags": self.pattern_tags,
            "matched_problem_id": self.matched_problem_id,
            "confidence": round(float(self.confidence), 4),
        }


class ASTInferenceEngine:
    """
    Loads precomputed dataset + embeddings once, then serves repeated
    `analyze_snippet` calls against them.

    Attributes:
        dataset: Parsed contents of data/unified_dataset.json. Expected shape
            is a list of records, each with at least an "id", "time_complexity",
            "space_complexity", and "pattern_tags" field, in the same order as
            the rows of `embeddings`.
        embeddings: (N, D) float32 array loaded from
            data/processed/problem_embeddings.npy, where row i corresponds to
            dataset[i].
    """

    def __init__(
        self,
        dataset_path: str = "data/unified_dataset.json",
        embeddings_path: str = "data/processed/problem_embeddings.npy",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.embeddings_path = Path(embeddings_path)

        self.dataset: list[dict[str, Any]] = self._load_dataset()
        self.embeddings: np.ndarray = self._load_embeddings()

        if len(self.dataset) != self.embeddings.shape[0]:
            raise ValueError(
                f"Dataset/embedding row mismatch: dataset has "
                f"{len(self.dataset)} records but embeddings has "
                f"{self.embeddings.shape[0]} rows. These must be aligned "
                f"1:1 by index."
            )

        self._parsers: dict[str, Any] = {}
        self._init_parsers()

        logger.info(
            "ASTInferenceEngine ready: %d reference records, embedding dim %d",
            len(self.dataset),
            self.embeddings.shape[1] if self.embeddings.ndim == 2 else -1,
        )

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #
    def _load_dataset(self) -> list[dict[str, Any]]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found at '{self.dataset_path}'. "
                f"Expected the unified dataset produced by the B-track "
                f"data pipeline."
            )
        try:
            with self.dataset_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Dataset file '{self.dataset_path}' is not valid JSON: {e}"
            ) from e

        if not isinstance(data, list):
            raise ValueError(
                f"Expected '{self.dataset_path}' to contain a JSON list of "
                f"records, got {type(data).__name__}."
            )
        return data

    def _load_embeddings(self) -> np.ndarray:
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found at '{self.embeddings_path}'. "
                f"Run the embedding-generation step of the pipeline first."
            )
        try:
            embeddings = np.load(self.embeddings_path)
        except (OSError, ValueError) as e:
            raise ValueError(
                f"Failed to load embeddings from '{self.embeddings_path}': {e}"
            ) from e
        return embeddings.astype(np.float32, copy=False)

    def _init_parsers(self) -> None:
        """Set up one tree-sitter parser per supported language."""
        if get_parser is None:
            raise ImportError(
                "tree-sitter-languages is not installed. Run "
                "`pip install tree-sitter tree-sitter-languages`."
            )
        for lang in SUPPORTED_LANGUAGES:
            try:
                self._parsers[lang] = get_parser(lang)
            except Exception as e:  # pragma: no cover
                logger.warning("Could not initialize parser for '%s': %s", lang, e)

    # ------------------------------------------------------------------ #
    # Core inference
    # ------------------------------------------------------------------ #
    def analyze_snippet(self, source_code: str, language: str) -> dict[str, Any]:
        """
        Analyze a source-code snippet and return its estimated complexity
        profile and matched pattern tags.

        Args:
            source_code: Raw source code to analyze.
            language: One of "python", "java", "c" (case-insensitive).

        Returns:
            A dict with keys: time_complexity, space_complexity,
            pattern_tags, matched_problem_id, confidence. On failure, returns
            a dict with an "error" key instead, so the Rust side can branch
            on that.
        """
        language = language.lower().strip()

        if language not in SUPPORTED_LANGUAGES:
            return {
                "error": f"Unsupported language '{language}'. "
                f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
            }
        if not source_code or not source_code.strip():
            return {"error": "source_code is empty."}

        try:
            tree = self._parse_to_ast(source_code, language)
            embedding = self._embed_ast(tree, source_code, language)
            result = self._match_against_dataset(embedding)
            return result.to_dict()
        except Exception as e:
            logger.exception("analyze_snippet failed")
            return {"error": f"Inference failed: {e}"}

    def _parse_to_ast(self, source_code: str, language: str):
        """Parse source into a tree-sitter AST for the given language."""
        parser = self._parsers.get(language)
        if parser is None:
            raise RuntimeError(f"No parser initialized for language '{language}'.")
        tree = parser.parse(bytes(source_code, "utf-8"))
        return tree

    def _embed_ast(self, tree, source_code: str, language: str) -> np.ndarray:
        """
        Generate a fixed-size embedding vector for the parsed AST.

        NOTE: This is the integration point for your actual embedding model
        (e.g. the sentence-embedding / AST-encoder work from B3). Replace the
        body below with a call into that model. The current implementation
        is a structural placeholder (node-type frequency vector) so the rest
        of the pipeline is runnable end-to-end before the real encoder is
        wired in.
        """
        # --- Placeholder: bag-of-node-types frequency vector ------------- #
        # Swap this out for your trained encoder's forward pass.
        node_type_counts: dict[str, int] = {}

        def walk(node):
            node_type_counts[node.type] = node_type_counts.get(node.type, 0) + 1
            for child in node.children:
                walk(child)

        walk(tree.root_node)

        target_dim = (
            self.embeddings.shape[1] if self.embeddings.ndim == 2 else 128
        )
        vec = np.zeros(target_dim, dtype=np.float32)
        for i, (_, count) in enumerate(node_type_counts.items()):
            vec[i % target_dim] += count

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _match_against_dataset(self, query_embedding: np.ndarray) -> MatchResult:
        """Cosine-similarity match against the precomputed embedding matrix."""
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] == 0:
            return MatchResult(
                time_complexity="unknown",
                space_complexity="unknown",
                pattern_tags=[],
            )

        norms = np.linalg.norm(self.embeddings, axis=1)
        norms[norms == 0] = 1e-8
        normalized = self.embeddings / norms[:, None]

        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            q_norm = 1e-8
        q = query_embedding / q_norm

        similarities = normalized @ q
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_record = self.dataset[best_idx]

        return MatchResult(
            time_complexity=best_record.get("time_complexity", "unknown"),
            space_complexity=best_record.get("space_complexity", "unknown"),
            pattern_tags=list(best_record.get("pattern_tags", [])),
            matched_problem_id=best_record.get("id"),
            confidence=best_score,
        )


# --------------------------------------------------------------------------- #
# Rust backend entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """
    Read a single JSON payload from stdin, run inference, write a single
    JSON object to stdout. Nothing else is ever written to stdout.

    Expected stdin payload:
        {"code": "<source code>", "language": "python|java|c"}

    Optional stdin fields:
        {"dataset_path": "...", "embeddings_path": "..."}  # override defaults
    """
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON on stdin: {e}"}))
        return

    code = payload.get("code")
    language = payload.get("language")

    if code is None or language is None:
        print(json.dumps({"error": "Payload must include 'code' and 'language'."}))
        return

    dataset_path = payload.get("dataset_path", "data/unified_dataset.json")
    embeddings_path = payload.get(
        "embeddings_path", "data/processed/problem_embeddings.npy"
    )

    try:
        engine = ASTInferenceEngine(
            dataset_path=dataset_path, embeddings_path=embeddings_path
        )
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(json.dumps({"error": f"Engine initialization failed: {e}"}))
        return

    result = engine.analyze_snippet(source_code=code, language=language)
    # Strictly one JSON object on stdout, nothing else.
    print(json.dumps(result))

if __name__ == "__main__":
    main()