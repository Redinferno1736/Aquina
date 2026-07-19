"""
inference.py

Inference wrapper for the Aquina YouTube Content-Type Classifier
(DistilBERT fine-tune: Coding=1, Entertainment=0).

Designed for two usage modes:

1. Direct import (if you ever run Python-side code / tests):

    from inference import YoutubeClassifier
    clf = YoutubeClassifier("models/youtube_classifier")
    result = clf.predict(title="Python in 100 Seconds",
                          channel="Fireship",
                          description="Learn Python basics fast.")
    # -> {"label": 1, "label_name": "Coding", "confidence": 0.98, "probs": {...}}

2. As a long-lived subprocess called from Rust (recommended):

   Rust spawns this script ONCE with `--serve`, keeping stdin/stdout open
   as pipes. The model loads a single time; after that every request is
   just a JSON line in -> JSON line out. This avoids paying the ~1-2s
   model-load cost on every call, which you would hit with a fresh
   `python inference.py` process per request.

   Request  (stdin,  one JSON object per line):
       {"id": "1", "title": "...", "channel": "...", "description": "..."}

   Response (stdout, one JSON object per line, same "id" echoed back):
       {"id": "1", "label": 1, "label_name": "Coding", "confidence": 0.98,
        "probs": {"entertainment": 0.02, "coding": 0.98}}

   On error for a given request, the response instead has:
       {"id": "1", "error": "..."}

   Rust side (conceptually, using e.g. `tokio::process::Command`):
       - spawn: python3 inference.py --serve --model-dir models/youtube_classifier
       - write one JSON line + "\n" to the child's stdin, flush
       - read one JSON line back from the child's stdout
       - keep the process alive and reuse it for all future requests

   You can also run a single one-off prediction from the CLI without
   --serve, useful for testing:

       python3 inference.py --model-dir models/youtube_classifier \\
           --title "Python in 100 Seconds" --channel "Fireship" \\
           --description "Learn Python basics fast."
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_NAMES = {0: "Entertainment", 1: "Coding"}
MAX_LENGTH = 256


class YoutubeClassifier:
    """
    Loads the fine-tuned DistilBERT youtube-content-type model once,
    and exposes predict() / predict_batch() for repeated inference.

    Instantiate this ONCE per process (model load is the expensive part).
    """

    def __init__(self, model_dir: str, device: Optional[str] = None):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir)
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def build_text(title: str, channel: str, description: str = "") -> str:
        """Must match the exact format used in preprocess.py at training time."""
        title = (title or "").strip()
        channel = (channel or "").strip()
        description = (description or "").strip()[:250]
        return f"[TITLE] {title} [CHANNEL] {channel} [DESC] {description}"

    @torch.no_grad()
    def predict(
        self,
        title: str,
        channel: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Run inference on a single video. Returns label, name, confidence, full probs."""
        text = self.build_text(title, channel, description)

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        label = int(torch.argmax(probs).item())
        confidence = float(probs[label].item())

        return {
            "label": label,
            "label_name": LABEL_NAMES.get(label, str(label)),
            "confidence": confidence,
            "probs": {
                "entertainment": float(probs[0].item()),
                "coding": float(probs[1].item()),
            },
        }

    @torch.no_grad()
    def predict_batch(self, items) -> list:
        """
        items: list of dicts with keys title/channel/description.
        Returns a list of prediction dicts in the same order.
        Batches through the model in one forward pass for efficiency.
        """
        texts = [
            self.build_text(
                it.get("title", ""), it.get("channel", ""), it.get("description", "")
            )
            for it in items
        ]

        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        results = []
        for row in probs:
            label = int(torch.argmax(row).item())
            results.append(
                {
                    "label": label,
                    "label_name": LABEL_NAMES.get(label, str(label)),
                    "confidence": float(row[label].item()),
                    "probs": {
                        "entertainment": float(row[0].item()),
                        "coding": float(row[1].item()),
                    },
                }
            )
        return results


def _serve_loop(clf: YoutubeClassifier) -> None:
    """
    Reads one JSON request per line from stdin, writes one JSON response
    per line to stdout, flushing after every response so the parent
    (Rust) process sees it immediately. Runs until stdin closes (EOF).
    """
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"error": f"invalid JSON: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        req_id = req.get("id")
        try:
            result = clf.predict(
                title=req.get("title", ""),
                channel=req.get("channel", ""),
                description=req.get("description", ""),
            )
            result["id"] = req_id
        except Exception as e:  # noqa: BLE001 — surface any failure to Rust, never crash the loop
            result = {"id": req_id, "error": str(e)}

        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Aquina YouTube classifier inference")
    parser.add_argument(
        "--model-dir",
        default="models/youtube_classifier",
        help="Path to the trained model directory (from train.py's OUTPUT_DIR).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as a long-lived process: read JSON requests from stdin, "
        "write JSON responses to stdout, one per line. Use this from Rust.",
    )
    parser.add_argument("--title", default="", help="One-off prediction: video title")
    parser.add_argument("--channel", default="", help="One-off prediction: channel name")
    parser.add_argument(
        "--description", default="", help="One-off prediction: video description"
    )

    args = parser.parse_args()

    clf = YoutubeClassifier(args.model_dir)

    if args.serve:
        _serve_loop(clf)
        return

    # One-off CLI prediction, prints a single JSON object to stdout.
    result = clf.predict(
        title=args.title, channel=args.channel, description=args.description
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()