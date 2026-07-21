"""
Aquina A3 - Head training on cached embeddings
Trains the language-embedding + regression head on the embeddings produced
by precompute_embeddings.py. This is fast (seconds per epoch) since the
expensive CodeBERT forward pass already happened once, upfront.

Run:
    python train_head.py --embeddings ./data/embeddings.pt --epochs 30
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

LANGUAGE_EMBED_DIM = 16
NUM_LANGUAGES = 6
BACKBONE_HIDDEN = 768
BACKBONE_NAME = "microsoft/codebert-base"
MAX_TOKEN_LEN = 512


class RegressionHead(nn.Module):
    """Same head architecture as model.py's ComplexityRegressor, but taking
    a precomputed embedding directly instead of running the backbone."""

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


def compute_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = math.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="./models")
    args = parser.parse_args()

    print("=" * 60)
    print("Aquina A3 - Head Training (on cached embeddings)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Using device: {device}")

    print(f"[setup] Loading cached embeddings from {args.embeddings} ...")
    cache = torch.load(args.embeddings)
    embeddings = cache["embeddings"]              # (N, 768)
    language_id = cache["language_id"]             # (N,)
    cc = cache["cyclomatic_complexity"]            # (N,)
    print(f"[setup] Loaded {embeddings.shape[0]} cached rows, embed dim {embeddings.shape[1]}")

    # log1p + standardize, same transform as before
    cc_log = torch.log1p(cc)
    target_mean = cc_log.mean().item()
    target_std = cc_log.std().item()
    cc_norm = (cc_log - target_mean) / target_std
    print(f"[setup] target_mean={target_mean:.4f}  target_std={target_std:.4f}")

    dataset = TensorDataset(embeddings, language_id, cc_norm)
    val_len = int(len(dataset) * args.val_size)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len],
                                     generator=torch.Generator().manual_seed(42))
    print(f"[setup] Train: {train_len}  Val: {val_len}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[setup] {len(train_loader)} train batches, {len(val_loader)} val batches")

    model = RegressionHead().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[setup] Head parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    print("=" * 60)
    print(f"Starting training: {args.epochs} epochs")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        train_bar = tqdm(train_loader, desc=f"[train] epoch {epoch}", unit="batch", leave=False)
        for emb, lang, target in train_bar:
            emb, lang, target = emb.to(device), lang.to(device), target.to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(emb, lang)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * emb.size(0)
            seen += emb.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / seen

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for emb, lang, target in val_loader:
                emb, lang = emb.to(device), lang.to(device)
                pred = model(emb, lang).cpu().squeeze(1)
                all_preds.extend(pred.tolist())
                all_targets.extend(target.tolist())

        metrics = compute_metrics(all_preds, all_targets)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_mae={metrics['mae']:.4f}  val_rmse={metrics['rmse']:.4f}  val_r2={metrics['r2']:.4f}")

        if metrics["mae"] < best_val_mae:
            best_val_mae = metrics["mae"]
            torch.save(model.state_dict(), out_dir / "head_best.pt")
            print(f"  -> new best checkpoint saved (val_mae={best_val_mae:.4f})")

    torch.save(model.state_dict(), out_dir / "head_final.pt")

    scaler_info = {
        "target_mean": target_mean,
        "target_std": target_std,
        "transform": "log1p_then_standardize",
        "language_to_id": {
            "python": 0, "java": 1, "javascript": 2, "php": 3, "ruby": 4, "go": 5,
        },
        "max_token_len": MAX_TOKEN_LEN,
        "backbone": BACKBONE_NAME,
        "lang_embed_dim": LANGUAGE_EMBED_DIM,
    }
    with open(out_dir / "scaler.json", "w") as f:
        json.dump(scaler_info, f, indent=2)

    print(f"\nDone. Best val MAE: {best_val_mae:.4f}")
    print(f"Checkpoints + scaler.json saved to {out_dir}")


if __name__ == "__main__":
    main()