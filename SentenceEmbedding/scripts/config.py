"""
config.py

Central configuration for the SentenceEmbedding project. All paths are
computed relative to this file's location so the project works
regardless of the current working directory it's launched from.
"""

from __future__ import annotations

import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

INDEX_DIR = os.path.join(PROJECT_ROOT, "index")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INDEX_FILE = os.path.join(INDEX_DIR, "aquina.index")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHUNK_TARGET_WORDS = 200
CHUNK_OVERLAP_WORDS = 40
MARKDOWN_MAX_SECTION_WORDS = 300

TOP_K = 10
PREVIEW_CHARS = 400

IGNORED_FOLDERS = {
    "__pycache__",
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".env",
    "index",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
}

IGNORED_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "aquina.index",
    "metadata.pkl",
}

IGNORED_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".exe",
    ".dll",
    ".so",
    ".bin",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".mp3",
    ".mp4",
    ".mov",
    ".index",
    ".pkl",
}

MARKDOWN_EXTENSIONS = {".md", ".markdown"}
CODE_EXTENSIONS = {".py"}
JSON_EXTENSIONS = {".json"}

os.makedirs(INDEX_DIR, exist_ok=True)