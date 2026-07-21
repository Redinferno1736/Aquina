"""
Aquina A3 - Inference
Class-based inference wrapper around the trained ComplexityRegressor.
Loads the frozen CodeBERT backbone + trained head (head_best.pt) + scaler.json
(including the per-language calibration), and exposes a simple predict() API.

This is the PyTorch reference implementation - meant to be run/tested locally
first. Once verified working, this same logic gets exported to ONNX for the
Rust/Tauri backend (separate step, after this is confirmed correct).

Usage as a script (smoke test):
    python inference.py --checkpoint ./models/head_best.pt --scaler ./models/scaler.json

Usage as a library (how your backend/other scripts will actually use it):
    from inference import ComplexityPredictor

    predictor = ComplexityPredictor(
        checkpoint_path="./models/head_best.pt",
        scaler_path="./models/scaler.json",
    )
    result = predictor.predict(code_string, language="python")
    print(result)  # {"raw_complexity": 4.2, "calibrated_complexity": 4.0, "language": "python"}
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


BACKBONE_NAME = "microsoft/codebert-base"
LANGUAGE_EMBED_DIM = 16
NUM_LANGUAGES = 6
BACKBONE_HIDDEN = 768


class RegressionHead(nn.Module):
    """Matches the architecture trained in train_head.py exactly."""

    def __init__(self, num_languages=NUM_LANGUAGES, lang_embed_dim=LANGUAGE_EMBED_DIM):
        super().__init__()
        self.lang_embedding = nn.Embedding(num_languages, lang_embed_dim)
        combined_dim = BACKBONE_HIDDEN + lang_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, pooled_embedding, language_id):
        lang_vec = self.lang_embedding(language_id)
        combined = torch.cat([pooled_embedding, lang_vec], dim=1)
        return self.mlp(combined)


class CodeEmbedder:
    """
    Wraps the frozen CodeBERT backbone + tokenizer + pooling logic.
    This is the same embedding pipeline used in precompute_embeddings.py -
    kept as its own class here since A4/A5 will need to reuse exactly this
    (tokenize -> backbone -> mean-pool) later for project-level fingerprints.
    """

    def __init__(self, backbone_name: str = BACKBONE_NAME, device: str = None, max_token_len: int = 512):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_token_len = max_token_len

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name).to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed(self, code: str) -> torch.Tensor:
        """Takes one code string, returns a single (768,) pooled embedding tensor on CPU."""
        enc = self.tokenizer(
            code,
            truncation=True,
            max_length=self.max_token_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeds = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        return pooled.squeeze(0).cpu()  # (768,)

    @torch.no_grad()
    def embed_batch(self, codes: list) -> torch.Tensor:
        """Takes a list of code strings, returns (N, 768) pooled embeddings on CPU."""
        enc = self.tokenizer(
            codes,
            truncation=True,
            max_length=self.max_token_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeds = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        return pooled.cpu()  # (N, 768)


class ComplexityPredictor:
    """
    Full end-to-end predictor: code + language -> predicted cyclomatic complexity.

    Loads:
      - the frozen CodeBERT backbone + tokenizer (via CodeEmbedder)
      - the trained regression head (head_best.pt)
      - scaler.json, which holds the log1p/standardize transform stats AND
        the per-language linear calibration fitted in calibrate.py

    predict() returns both the raw (uncalibrated) and calibrated complexity,
    so callers can compare or fall back to raw if they ever want to.
    """

    def __init__(self, checkpoint_path: str, scaler_path: str, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        with open(scaler_path, "r") as f:
            self.scaler = json.load(f)

        self.language_to_id = self.scaler["language_to_id"]
        self.target_mean = self.scaler["target_mean"]
        self.target_std = self.scaler["target_std"]
        self.calibration = self.scaler.get("per_language_calibration", {})
        max_token_len = self.scaler.get("max_token_len", 512)
        backbone_name = self.scaler.get("backbone", BACKBONE_NAME)

        self.embedder = CodeEmbedder(backbone_name=backbone_name, device=str(self.device),
                                      max_token_len=max_token_len)

        self.head = RegressionHead().to(self.device)
        self.head.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.head.eval()

    def _invert_transform(self, normalized_value: float) -> float:
        """Undoes the log1p + standardize transform to get back a real complexity number."""
        log_value = normalized_value * self.target_std + self.target_mean
        return float(np.expm1(log_value))

    def _apply_calibration(self, normalized_pred: float, language: str) -> float:
        """Applies the per-language linear correction fitted in calibrate.py, if present."""
        cal = self.calibration.get(language)
        if cal is None:
            return normalized_pred
        return cal["a"] * normalized_pred + cal["b"]

    @torch.no_grad()
    def predict(self, code: str, language: str) -> dict:
        """
        Predicts cyclomatic complexity for a single function.

        Args:
            code: the function's source code as a string
            language: one of "python", "java", "javascript", "php", "ruby", "go"

        Returns:
            dict with raw_complexity, calibrated_complexity, and the language used
        """
        language = language.lower()
        if language not in self.language_to_id:
            raise ValueError(f"Unknown language '{language}'. Must be one of {list(self.language_to_id)}")

        lang_id = self.language_to_id[language]

        pooled_embedding = self.embedder.embed(code).unsqueeze(0).to(self.device)   # (1, 768)
        lang_tensor = torch.tensor([lang_id], dtype=torch.long).to(self.device)

        pred_normalized = self.head(pooled_embedding, lang_tensor).item()

        raw_complexity = self._invert_transform(pred_normalized)

        calibrated_normalized = self._apply_calibration(pred_normalized, language)
        calibrated_complexity = self._invert_transform(calibrated_normalized)

        return {
            "language": language,
            "raw_complexity": round(raw_complexity, 2),
            "calibrated_complexity": round(calibrated_complexity, 2),
        }

    @torch.no_grad()
    def predict_batch(self, codes: list, languages: list) -> list:
        """Same as predict(), but for a list of (code, language) pairs at once - more
        efficient than calling predict() in a loop when scoring many functions."""
        if len(codes) != len(languages):
            raise ValueError("codes and languages must be the same length")

        pooled_embeddings = self.embedder.embed_batch(codes).to(self.device)  # (N, 768)
        lang_ids = [self.language_to_id[lang.lower()] for lang in languages]
        lang_tensor = torch.tensor(lang_ids, dtype=torch.long).to(self.device)

        preds_normalized = self.head(pooled_embeddings, lang_tensor).squeeze(1).tolist()

        results = []
        for pred_norm, language in zip(preds_normalized, languages):
            language = language.lower()
            raw = self._invert_transform(pred_norm)
            calibrated_norm = self._apply_calibration(pred_norm, language)
            calibrated = self._invert_transform(calibrated_norm)
            results.append({
                "language": language,
                "raw_complexity": round(raw, 2),
                "calibrated_complexity": round(calibrated, 2),
            })
        return results


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the ComplexityPredictor")
    parser.add_argument("--checkpoint", type=str, default="./models/head_best.pt")
    parser.add_argument("--scaler", type=str, default="./models/scaler.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Aquina A3 - Inference smoke test")
    print("=" * 60)

    print("[setup] Loading predictor (backbone + head + scaler) ...")
    predictor = ComplexityPredictor(checkpoint_path=args.checkpoint, scaler_path=args.scaler)
    print("[setup] Predictor ready\n")

    test_cases = [
        {
            "language": "python",
            "code": "def add(a, b):\n    return a + b",
        },
        {
            "language": "python",
            "code": (
                "def classify(value, user_tier, coupon_code=None, is_holiday=False):\n"
                "    if user_tier == 'premium':\n"
                "        discount = 0.2\n"
                "    elif user_tier == 'standard':\n"
                "        discount = 0.1\n"
                "    else:\n"
                "        discount = 0\n"
                "    if coupon_code:\n"
                "        discount += 0.05\n"
                "    if is_holiday:\n"
                "        for i in range(3):\n"
                "            if i % 2 == 0:\n"
                "                discount += 0.01\n"
                "    return value * (1 - discount)"
            ),
        },
        {
            "language": "java",
            "code": (
                "public int add(int a, int b) {\n"
                "    return a + b;\n"
                "}"
            ),
        },
        {
            "language": "javascript",
            "code": (
                "function processItems(items) {\n"
                "  let result = [];\n"
                "  for (let i = 0; i < items.length; i++) {\n"
                "    if (items[i].active) {\n"
                "      if (items[i].value > 100) {\n"
                "        result.push(items[i].value * 2);\n"
                "      } else {\n"
                "        result.push(items[i].value);\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "  return result;\n"
                "}"
            ),
        },
    ]

    print("[test] Running predictions on sample snippets:\n")
    for i, case in enumerate(test_cases, 1):
        result = predictor.predict(case["code"], case["language"])
        print(f"  [{i}] language={result['language']:<12} "
              f"raw={result['raw_complexity']:.2f}  calibrated={result['calibrated_complexity']:.2f}")

    print("\n[test] Running the same 4 snippets through predict_batch() for consistency check ...")
    codes = [c["code"] for c in test_cases]
    langs = [c["language"] for c in test_cases]
    batch_results = predictor.predict_batch(codes, langs)
    for i, result in enumerate(batch_results, 1):
        print(f"  [{i}] language={result['language']:<12} "
              f"raw={result['raw_complexity']:.2f}  calibrated={result['calibrated_complexity']:.2f}")

    print("\n[done] If both runs (single vs batch) show matching numbers per snippet, "
          "and simple functions score lower than the nested/branching ones, "
          "the inference pipeline is working correctly.")


if __name__ == "__main__":
    main()