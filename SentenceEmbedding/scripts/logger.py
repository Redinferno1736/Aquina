"""
logger.py

Shared logging configuration for the SentenceEmbedding project. Every
other module imports `logger` from here so log output is consistent
and configured in exactly one place.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _build_logger() -> logging.Logger:
    log = logging.getLogger("aquina")
    log.setLevel(logging.INFO)

    if not log.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        log.addHandler(handler)
        log.propagate = False

    return log


logger = _build_logger()