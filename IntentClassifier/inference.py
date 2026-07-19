"""
inference.py

Production inference pipeline for the Aquina intent classifier.

Pipeline:
    User Sentence
        -> Intent Prediction        (predictor.py :: AquinaIntentPredictor)
        -> Confidence Threshold Check
        -> Entity Extraction         (rule-based, per-intent helper functions)
        -> Entity Normalization      (alias dictionaries)
        -> Structured JSON Output

This file does NOT train or fine-tune anything. It only loads the already
trained model (models/aquina_intent_model/) and label encoder
(models/label_encoder.pkl) via predictor.py, and runs inference.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from predictor import AquinaIntentPredictor

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("aquina.inference")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

DEFAULT_MODEL_DIR = Path("models/aquina_intent_model")
DEFAULT_LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

@dataclass
class IntentResult:
    """Structured result of the full inference pipeline for one sentence."""

    intent: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ----------------------------------------------------------------------
# Loading the predictor
# ----------------------------------------------------------------------

def load_predictor(
    model_dir: Path = DEFAULT_MODEL_DIR,
    label_encoder_path: Path = DEFAULT_LABEL_ENCODER_PATH,
) -> AquinaIntentPredictor:
    """
    Load the already-trained AquinaIntentPredictor.

    Does NOT train or fine-tune anything — this only wires together the
    existing model directory and label encoder via predictor.py.

    Raises:
        FileNotFoundError: if the model directory or label encoder file
            is missing.
        Exception: re-raised after logging, for any other load failure.
    """
    model_dir = Path(model_dir)
    label_encoder_path = Path(label_encoder_path)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")

    try:
        with label_encoder_path.open("rb") as f:
            label_encoder = pickle.load(f)

        predictor = AquinaIntentPredictor(
            model_dir=str(model_dir),
            label_encoder=label_encoder,
        )
        logger.info("Loaded AquinaIntentPredictor (model_dir=%s)", model_dir)
        return predictor

    except Exception:
        logger.exception("Failed to load predictor from %s / %s", model_dir, label_encoder_path)
        raise


# ----------------------------------------------------------------------
# Text-cleaning helpers (shared by every extractor, to avoid repeating regex)
# ----------------------------------------------------------------------

_LEADING_POLITENESS = re.compile(
    r"^\s*(hey|yo|bro|so|um|uh|okay|ok|actually)?[,]?\s*"
    r"(please|can you|could you|would you|kindly|pls|plz)?\s*",
    re.IGNORECASE,
)
_TRAILING_FILLER = re.compile(
    r"\s*(please|now|asap|for me|if possible|when you can|thanks|ty|"
    r"right now|real quick|lol)?\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def strip_filler(text: str) -> str:
    """Remove common leading politeness phrases and trailing filler/punctuation."""
    cleaned = _LEADING_POLITENESS.sub("", text)
    cleaned = _TRAILING_FILLER.sub("", cleaned)
    return cleaned.strip()


def strip_leading_verbs(text: str, verbs: list[str]) -> str:
    """
    Remove a leading verb phrase (e.g. 'open', 'launch', 'fire up') from the
    start of an already-filler-stripped sentence, returning what remains
    as the raw entity text.
    """
    pattern = re.compile(r"^(" + "|".join(re.escape(v) for v in verbs) + r")\s+", re.IGNORECASE)
    return pattern.sub("", text, count=1).strip()


def extract_number(text: str) -> Optional[int]:
    """Return the first integer found in the text, if any."""
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def extract_direction(text: str) -> Optional[str]:
    """Return 'up' or 'down' based on keywords, if no explicit number is present."""
    lowered = text.lower()
    if any(w in lowered for w in ("up", "increase", "louder", "brighter", "raise")):
        return "up"
    if any(w in lowered for w in ("down", "decrease", "lower", "quieter", "dim", "reduce")):
        return "down"
    return None


# ----------------------------------------------------------------------
# Normalization alias dictionaries
# ----------------------------------------------------------------------

APP_ALIASES: Dict[str, str] = {
    "chrome": "Google Chrome",
    "google chrome": "Google Chrome",
    "chrome browser": "Google Chrome",
    "gchrome": "Google Chrome",
    "chorme": "Google Chrome",
    "vs code": "Visual Studio Code",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "visual studio code": "Visual Studio Code",
    "vsc": "Visual Studio Code",
    "vs": "Visual Studio Code",
    "spotify": "Spotify",
    "spotfy": "Spotify",
    "discord": "Discord",
    "photoshop": "Adobe Photoshop",
    "ps": "Adobe Photoshop",
    "adobe ps": "Adobe Photoshop",
    "photo editor": "Adobe Photoshop",
    "powerpoint": "Microsoft PowerPoint",
    "ppt": "Microsoft PowerPoint",
    "excel": "Microsoft Excel",
    "word": "Microsoft Word",
    "explorer": "File Explorer",
    "file explorer": "File Explorer",
    "terminal": "Terminal",
    "term": "Terminal",
    "cmd": "Command Prompt",
    "android studio": "Android Studio",
    "studio": "Android Studio",
}

FOLDER_ALIASES: Dict[str, str] = {
    "downloads": "Downloads",
    "documents": "Documents",
    "desktop": "Desktop",
    "pictures": "Pictures",
    "videos": "Videos",
    "music": "Music",
}

SEARCH_ENGINE_ALIASES: Dict[str, str] = {
    "youtube": "YouTube",
    "yt": "YouTube",
    "google": "Google",
}


def _normalize(value: str, alias_map: Dict[str, str]) -> str:
    """
    Normalize an extracted entity string using an alias dictionary.
    Falls back to a title-cased version of the original value if no
    alias is found, so unseen apps/folders still get a clean display form.
    """
    key = value.strip().lower()
    if key in alias_map:
        return alias_map[key]
    return value.strip().title() if value.strip().islower() else value.strip()


def normalize_app(name: str) -> str:
    return _normalize(name, APP_ALIASES)


def normalize_folder(name: str) -> str:
    return _normalize(name, FOLDER_ALIASES)


# ----------------------------------------------------------------------
# Per-intent entity extractors
# ----------------------------------------------------------------------

_OPEN_VERBS = ["open", "launch", "start", "run", "boot", "fire up", "bring up",
               "open up", "get", "switch to", "let's open", "spin up", "load",
               "pull up", "i need", "i'd like to open", "i want to use"]
_CLOSE_VERBS = ["close", "quit", "exit", "terminate", "kill", "stop",
                "shut down", "close down", "shut", "end", "get rid of"]


def extract_open_app(sentence: str) -> Dict[str, Any]:
    """'open Chrome' -> {'app': 'Google Chrome'}"""
    cleaned = strip_filler(sentence)
    raw_app = strip_leading_verbs(cleaned, _OPEN_VERBS)
    if not raw_app:
        return {}
    return {"app": normalize_app(raw_app)}


def extract_close_app(sentence: str) -> Dict[str, Any]:
    """'close VS Code' -> {'app': 'Visual Studio Code'}"""
    cleaned = strip_filler(sentence)
    raw_app = strip_leading_verbs(cleaned, _CLOSE_VERBS)
    if not raw_app:
        return {}
    return {"app": normalize_app(raw_app)}


def extract_create_folder(sentence: str) -> Dict[str, Any]:
    """'create folder Projects' -> {'folder_name': 'Projects'}"""
    cleaned = strip_filler(sentence)
    match = re.search(
        r"(?:create|make|new)\s+(?:a\s+)?(?:folder|directory)\s*(?:called|named)?\s+(.+)",
        cleaned,
        re.IGNORECASE,
    )
    if not match:
        return {}
    return {"folder_name": normalize_folder(match.group(1))}


def extract_delete_file(sentence: str) -> Dict[str, Any]:
    """'delete Downloads/test.pdf' -> {'path': 'Downloads/test.pdf'}"""
    cleaned = strip_filler(sentence)
    raw_path = strip_leading_verbs(
        cleaned, ["delete", "remove", "trash", "get rid of the file", "get rid of"]
    )
    raw_path = re.sub(r"^the\s+file\s+", "", raw_path, flags=re.IGNORECASE).strip()
    if not raw_path:
        return {}
    return {"path": raw_path}


def extract_rename_file(sentence: str) -> Dict[str, Any]:
    """'rename notes.txt to todo.txt' -> {'old_name': 'notes.txt', 'new_name': 'todo.txt'}"""
    cleaned = strip_filler(sentence)
    match = re.search(
        r"rename\s+(?:the\s+file\s+)?(.+?)\s+to\s+(.+)",
        cleaned,
        re.IGNORECASE,
    )
    if not match:
        return {}
    return {"old_name": match.group(1).strip(), "new_name": match.group(2).strip()}


def extract_copy_file(sentence: str) -> Dict[str, Any]:
    """'copy image.png to Desktop' -> {'source': 'image.png', 'destination': 'Desktop'}"""
    cleaned = strip_filler(sentence)
    match = re.search(
        r"(?:copy|duplicate)\s+(?:a\s+copy\s+of\s+)?(.+?)\s+to\s+(.+)",
        cleaned,
        re.IGNORECASE,
    )
    if not match:
        return {}
    return {
        "source": match.group(1).strip(),
        "destination": normalize_folder(match.group(2).strip()),
    }


def extract_move_file(sentence: str) -> Dict[str, Any]:
    """'move report.pdf to Documents' -> {'source': 'report.pdf', 'destination': 'Documents'}"""
    cleaned = strip_filler(sentence)
    match = re.search(r"move\s+(.+?)\s+to\s+(.+)", cleaned, re.IGNORECASE)
    if not match:
        return {}
    return {
        "source": match.group(1).strip(),
        "destination": normalize_folder(match.group(2).strip()),
    }


def extract_search_web(sentence: str) -> Dict[str, Any]:
    """'search python decorators' -> {'query': 'python decorators'}"""
    cleaned = strip_filler(sentence)
    query = re.sub(
        r"^(search( the web)?( for)?|google|look up)\s+", "", cleaned, flags=re.IGNORECASE
    ).strip()
    if not query:
        return {}
    return {"query": query}


def extract_youtube_search(sentence: str) -> Dict[str, Any]:
    """'search cute cats on YouTube' -> {'query': 'cute cats'}"""
    cleaned = strip_filler(sentence)
    query = re.sub(
        r"^(search( for)?|find|play)\s+", "", cleaned, flags=re.IGNORECASE
    ).strip()
    query = re.sub(
        r"\s+on\s+(youtube|yt)\s*$", "", query, flags=re.IGNORECASE
    ).strip()
    if not query:
        return {}
    return {"query": query}


def extract_set_volume(sentence: str) -> Dict[str, Any]:
    """
    'set volume to 40' -> {'value': 40}
    'increase volume'   -> {'direction': 'up'}
    'decrease volume'   -> {'direction': 'down'}
    """
    cleaned = strip_filler(sentence)
    value = extract_number(cleaned)
    if value is not None:
        return {"value": value}
    direction = extract_direction(cleaned)
    if direction:
        return {"direction": direction}
    return {}


def extract_set_brightness(sentence: str) -> Dict[str, Any]:
    """
    'set brightness to 75' -> {'value': 75}
    'increase brightness'  -> {'direction': 'up'}
    """
    cleaned = strip_filler(sentence)
    value = extract_number(cleaned)
    if value is not None:
        return {"value": value}
    direction = extract_direction(cleaned)
    if direction:
        return {"direction": direction}
    return {}


def extract_open_project(sentence: str) -> Dict[str, Any]:
    """'open project Aquina' -> {'project': 'Aquina'}"""
    cleaned = strip_filler(sentence)
    match = re.search(
        r"(?:open|load|launch|start)\s+(?:my\s+)?(?:project\s+)?(.+?)(?:\s+project)?$",
        cleaned,
        re.IGNORECASE,
    )
    if not match:
        return {}
    return {"project": match.group(1).strip()}


def extract_run_command(sentence: str) -> Dict[str, Any]:
    """'run git pull' -> {'command': 'git pull'}"""
    cleaned = strip_filler(sentence)
    command = re.sub(
        r"^(run|execute)( command)?\s+", "", cleaned, flags=re.IGNORECASE
    ).strip()
    if not command:
        return {}
    return {"command": command}


def extract_no_entities(_: str) -> Dict[str, Any]:
    """For intents that carry no entity payload (e.g. shutdown_pc, mute)."""
    return {}


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

ENTITY_EXTRACTORS: Dict[str, Callable[[str], Dict[str, Any]]] = {
    "open_app": extract_open_app,
    "close_app": extract_close_app,
    "create_folder": extract_create_folder,
    "delete_file": extract_delete_file,
    "delete_folder": extract_create_folder,  # same 'name after keyword' shape
    "rename_file": extract_rename_file,
    "copy_file": extract_copy_file,
    "move_file": extract_move_file,
    "search_web": extract_search_web,
    "youtube_search": extract_youtube_search,
    "volume_set": extract_set_volume,
    "set_volume": extract_set_volume,
    "volume_up": lambda s: {"direction": "up"},
    "volume_down": lambda s: {"direction": "down"},
    "brightness_set": extract_set_brightness,
    "set_brightness": extract_set_brightness,
    "brightness_up": lambda s: {"direction": "up"},
    "brightness_down": lambda s: {"direction": "down"},
    "open_project": extract_open_project,
    "run_command": extract_run_command,
    "mute": extract_no_entities,
    "unmute": extract_no_entities,
    "shutdown_pc": extract_no_entities,
    "restart_pc": extract_no_entities,
    "sleep_pc": extract_no_entities,
    "lock_pc": extract_no_entities,
    "logout": extract_no_entities,
    "general_chat": extract_no_entities,
    "unknown": extract_no_entities,
}


def extract_entities(sentence: str, intent: str) -> Dict[str, Any]:
    """
    Dispatch to the correct entity extractor for the given predicted intent.
    Unknown/unhandled intents return an empty entity dict rather than raising,
    so the pipeline degrades gracefully instead of crashing on new intents.
    """
    extractor = ENTITY_EXTRACTORS.get(intent, extract_no_entities)
    try:
        return extractor(sentence)
    except Exception:
        logger.exception("Entity extraction failed for intent=%s sentence=%r", intent, sentence)
        return {}


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------

def run_inference(
    predictor: AquinaIntentPredictor,
    sentence: str,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> IntentResult:
    """
    Run the full inference pipeline for a single sentence:
    predict intent -> confidence check -> entity extraction -> structured result.
    """
    try:
        intent, confidence = predictor.predict(sentence)
    except Exception:
        logger.exception("Intent prediction failed for sentence=%r", sentence)
        return IntentResult(intent="unknown", confidence=0.0, entities={})

    if confidence < threshold:
        logger.info(
            "Confidence %.3f below threshold %.2f for sentence=%r -> returning 'unknown'",
            confidence, threshold, sentence,
        )
        return IntentResult(intent="unknown", confidence=confidence, entities={})

    entities = extract_entities(sentence, intent)
    return IntentResult(intent=intent, confidence=confidence, entities=entities)


# ----------------------------------------------------------------------
# Manual test run
# ----------------------------------------------------------------------

if __name__ == "__main__":
    example_commands = [
        "Open Chrome",
        "Launch Spotify",
        "Create folder Projects",
        "Delete test.txt",
        "Rename notes.txt to todo.txt",
        "Search Python tutorials",
        "Search lofi music on YouTube",
        "Increase brightness",
        "Set brightness to 80",
        "Decrease volume",
        "Set volume to 50",
        "Open Downloads",
        "Move report.pdf to Documents",
        "Copy image.png to Desktop",
    ]

    try:
        aquina_predictor = load_predictor()
    except FileNotFoundError as e:
        logger.error(
            "Could not load model/label encoder (%s). "
            "Make sure models/aquina_intent_model/ and models/label_encoder.pkl exist.",
            e,
        )
        raise SystemExit(1)

    for command in example_commands:
        result = run_inference(aquina_predictor, command)
        print(f"\nInput: {command}")
        print(result.to_json())