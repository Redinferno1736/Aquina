"""
Aquina A3 - Per-language calibration
Fits a simple linear correction (corrected = a * predicted + b) per language,
using the TRAIN split (not validation - fitting on val would leak eval data
into the correction itself and make the eval numbers dishonest).

This doesn't retrain anything - it's a cheap post-hoc fix for the Python
gap found in eval_per_language.py (R2=0.78 vs 0.84-0.95 for other languages).

Run:
    python calibrate.py --embeddings ./data/embeddings.pt --checkpoint ./models/head_best.pt --scaler ./models/scaler.json
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

LANG_ID_TO_NAME = {0: "python", 1: "java", 2: "javascript", 3: "php", 4: "ruby", 5: "go"}
LANGUAGE_EMBED_DIM = 16
NUM_LANGUAGES = 6
BACKBONE_HIDDEN = 768


class RegressionHead(nn.Module):
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


def fit_linear(preds, targets):
    """Fits corrected = a*preds + b via least squares. Returns (a, b)."""
    preds = np.array(preds)
    targets = np.array(targets)
    A = np.vstack([preds, np.ones_like(preds)]).T
    a, b = np.linalg.lstsq(A, targets, rcond=None)[0]
    return float(a), float(b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--scaler", type=str, required=True)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Using device: {device}")

    cache = torch.load(args.embeddings)
    embeddings = cache["embeddings"]
    language_id = cache["language_id"]
    cc = cache["cyclomatic_complexity"]

    cc_log = torch.log1p(cc)
    target_mean = cc_log.mean().item()
    target_std = cc_log.std().item()
    cc_norm = (cc_log - target_mean) / target_std

    dataset = TensorDataset(embeddings, language_id, cc_norm)
    val_len = int(len(dataset) * args.val_size)
    train_len = len(dataset) - val_len
    # Same seed as train_head.py - this gives us the TRAIN split, deliberately
    # not the val split, so calibration doesn't leak eval data into itself.
    train_ds, _ = random_split(dataset, [train_len, val_len],
                                generator=torch.Generator().manual_seed(42))
    print(f"[setup] Fitting calibration on train split: {len(train_ds)} rows")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

    model = RegressionHead().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"[setup] Loaded checkpoint from {args.checkpoint}")

    preds_by_lang = {i: [] for i in range(NUM_LANGUAGES)}
    targets_by_lang = {i: [] for i in range(NUM_LANGUAGES)}

    print("[calibrate] Running forward pass on train split ...")
    with torch.no_grad():
        for emb, lang, target in loader:
            emb, lang = emb.to(device), lang.to(device)
            pred = model(emb, lang).cpu().squeeze(1)
            for p, t, l in zip(pred.tolist(), target.tolist(), lang.cpu().tolist()):
                preds_by_lang[l].append(p)
                targets_by_lang[l].append(t)

    print("\n" + "=" * 60)
    print("Fitted per-language calibration (corrected = a * pred + b)")
    print("=" * 60)
    calibration = {}
    for lang_id in range(NUM_LANGUAGES):
        p, t = preds_by_lang[lang_id], targets_by_lang[lang_id]
        a, b = fit_linear(p, t)
        calibration[LANG_ID_TO_NAME[lang_id]] = {"a": a, "b": b}
        print(f"{LANG_ID_TO_NAME[lang_id]:<12} a={a:.4f}  b={b:.4f}  (n={len(p)})")

    # Load existing scaler.json and add the calibration block to it
    with open(args.scaler, "r") as f:
        scaler_info = json.load(f)

    scaler_info["per_language_calibration"] = calibration
    scaler_info["calibration_note"] = (
        "Apply as: corrected_normalized_pred = a * raw_normalized_pred + b, "
        "using the coefficients for the input's language, BEFORE inverting "
        "the log1p/standardize transform."
    )

    with open(args.scaler, "w") as f:
        json.dump(scaler_info, f, indent=2)

    print(f"\n[done] Calibration saved into {args.scaler}")


if __name__ == "__main__":
    main()