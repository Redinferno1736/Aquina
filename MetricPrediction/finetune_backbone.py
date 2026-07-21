"""
Aquina A3 - Backbone fine-tuning (targeted fix for Python's weaker R2)
Unfreezes the last N CodeBERT layers and fine-tunes on a SUBSAMPLE of the
full dataset, biased toward Python, since that's the language with the
measured gap (R2=0.78 vs 0.84-0.95 for others).

This is NOT run on the cached embeddings from precompute_embeddings.py -
those were computed with the OLD frozen backbone weights and become stale
the moment the backbone changes. This script re-reads raw code from the
parquet file and re-runs the (now partially trainable) backbone each batch.

Given 4GB VRAM, this is scoped deliberately small:
- Subsampled rows (default 60k, weighted toward python) instead of all 348k
- Shorter max_token_len (256 instead of 512) - most functions don't need 512
- Small number of epochs (default 3)
- Separate, low learning rate for the unfrozen backbone layers vs the head

Run:
    python finetune_backbone.py --data ./data/labeled_functions.parquet --epochs 3 --unfreeze_last_n 2
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BACKBONE_NAME = "microsoft/codebert-base"
LANGUAGE_EMBED_DIM = 16
NUM_LANGUAGES = 6
BACKBONE_HIDDEN = 768
LANG_ID_TO_NAME = {0: "python", 1: "java", 2: "javascript", 3: "php", 4: "ruby", 5: "go"}


class ComplexityRegressor(nn.Module):
    def __init__(self, num_languages=NUM_LANGUAGES, lang_embed_dim=LANGUAGE_EMBED_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(BACKBONE_NAME)
        self.lang_embedding = nn.Embedding(num_languages, lang_embed_dim)
        combined_dim = BACKBONE_HIDDEN + lang_embed_dim
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n(self, n):
        self.freeze_all_backbone()
        layers = self.backbone.encoder.layer
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask, language_id):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        token_embeds = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        lang_vec = self.lang_embedding(language_id)
        combined = torch.cat([pooled, lang_vec], dim=1)
        return self.head(combined)


class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, max_token_len):
        self.codes = df["code"].tolist()
        self.lang_ids = df["language_id"].tolist()
        self.targets = df["cc_normalized"].tolist()
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx], truncation=True, max_length=self.max_token_len,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "language_id": torch.tensor(self.lang_ids[idx], dtype=torch.long),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }


def compute_metrics(preds, targets):
    preds, targets = np.array(preds), np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = math.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def subsample_biased_toward_python(df, total_rows, python_fraction=0.4):
    """
    Builds a subsample where Python gets `python_fraction` of the rows and
    the other 5 languages split the remainder evenly - so fine-tuning
    concentrates on fixing Python without forgetting the other languages.
    """
    python_n = int(total_rows * python_fraction)
    other_n_each = int((total_rows - python_n) / 5)

    parts = []
    python_df = df[df["language"] == "python"]
    parts.append(python_df.sample(n=min(python_n, len(python_df)), random_state=42))

    for lang in ["java", "javascript", "php", "ruby", "go"]:
        lang_df = df[df["language"] == lang]
        parts.append(lang_df.sample(n=min(other_n_each, len(lang_df)), random_state=42))

    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--init_head", type=str, default="./models/head_best.pt",
                         help="Path to the already-trained head weights to warm-start from")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_token_len", type=int, default=256)
    parser.add_argument("--unfreeze_last_n", type=int, default=2)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--subsample_rows", type=int, default=60000)
    parser.add_argument("--python_fraction", type=float, default=0.4)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="./models")
    args = parser.parse_args()

    print("=" * 60)
    print("Aquina A3 - Backbone Fine-tuning (Python-focused)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Using device: {device}")
    if device.type == "cuda":
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")

    print(f"[setup] Loading full data from {args.data} ...")
    df = pd.read_parquet(args.data)
    print(f"[setup] Full dataset: {len(df)} rows")

    print(f"[setup] Subsampling to {args.subsample_rows} rows "
          f"({args.python_fraction:.0%} python, rest split across other 5 languages) ...")
    df = subsample_biased_toward_python(df, args.subsample_rows, args.python_fraction)
    print(f"[setup] Subsample distribution:\n{df['language'].value_counts()}")

    cc_log = np.log1p(df["cyclomatic_complexity"])
    target_mean = cc_log.mean()
    target_std = cc_log.std()
    df["cc_normalized"] = (cc_log - target_mean) / target_std
    print(f"[setup] target_mean={target_mean:.4f}  target_std={target_std:.4f}")
    print("[setup] NOTE: reusing the SAME transform stats convention as before "
          "for consistency with the existing scaler.json")

    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=42)
    print(f"[setup] Train: {len(train_df)}  Val: {len(val_df)}")

    print(f"[setup] Loading tokenizer + backbone {BACKBONE_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)

    train_ds = CodeDataset(train_df, tokenizer, args.max_token_len)
    val_ds = CodeDataset(val_df, tokenizer, args.max_token_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"[setup] {len(train_loader)} train batches, {len(val_loader)} val batches "
          f"(batch_size={args.batch_size}, max_token_len={args.max_token_len})")

    model = ComplexityRegressor().to(device)

    if Path(args.init_head).exists():
        print(f"[setup] Warm-starting head + language embedding from {args.init_head}")
        head_state = torch.load(args.init_head, map_location=device)
        model.lang_embedding.load_state_dict(
            {k.replace("lang_embedding.", ""): v for k, v in head_state.items() if k.startswith("lang_embedding.")}
        )
        model.head.load_state_dict(
            {k.replace("mlp.", ""): v for k, v in head_state.items() if k.startswith("mlp.")}
        )
    else:
        print(f"[setup] No existing head checkpoint found at {args.init_head}, starting head from scratch")

    model.unfreeze_last_n(args.unfreeze_last_n)
    print(f"[setup] Unfroze last {args.unfreeze_last_n} backbone layers")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.head.parameters()) + list(model.lang_embedding.parameters())
    num_backbone_trainable = sum(p.numel() for p in backbone_params)
    num_head_trainable = sum(p.numel() for p in head_params)
    print(f"[setup] Trainable backbone params: {num_backbone_trainable:,}")
    print(f"[setup] Trainable head params: {num_head_trainable:,}")

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ])
    loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float("inf")
    print("=" * 60)
    print(f"Starting fine-tuning: {args.epochs} epochs")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        train_bar = tqdm(train_loader, desc=f"[finetune] epoch {epoch}", unit="batch")
        for batch in train_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            language_id = batch["language_id"].to(device)
            target = batch["target"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            pred = model(input_ids, attention_mask, language_id)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)
            seen += input_ids.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss/seen:.4f}")

        train_loss = running_loss / seen
        print(f"[finetune] epoch {epoch} done - avg train_loss={train_loss:.4f}")

        model.eval()
        preds_by_lang = {i: [] for i in range(NUM_LANGUAGES)}
        targets_by_lang = {i: [] for i in range(NUM_LANGUAGES)}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[val] epoch {epoch}", unit="batch"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                language_id = batch["language_id"].to(device)
                target = batch["target"]

                pred = model(input_ids, attention_mask, language_id).cpu().squeeze(1)
                for p, t, l in zip(pred.tolist(), target.tolist(), batch["language_id"].tolist()):
                    preds_by_lang[l].append(p)
                    targets_by_lang[l].append(t)

        all_preds = [p for v in preds_by_lang.values() for p in v]
        all_targets = [t for v in targets_by_lang.values() for t in v]
        overall = compute_metrics(all_preds, all_targets)
        print(f"Epoch {epoch}/{args.epochs} OVERALL  train_loss={train_loss:.4f}  "
              f"val_mae={overall['mae']:.4f}  val_r2={overall['r2']:.4f}")
        for lang_id in range(NUM_LANGUAGES):
            p, t = preds_by_lang[lang_id], targets_by_lang[lang_id]
            if not p:
                continue
            m = compute_metrics(p, t)
            print(f"    {LANG_ID_TO_NAME[lang_id]:<12} n={m['mae']:.4f} mae  r2={m['r2']:.4f}  (n={len(p)})")

        if overall["mae"] < best_val_mae:
            best_val_mae = overall["mae"]
            torch.save(model.state_dict(), out_dir / "finetuned_model_best.pt")
            print(f"  -> new best full-model checkpoint saved (val_mae={best_val_mae:.4f})")

    torch.save(model.state_dict(), out_dir / "finetuned_model_final.pt")
    print(f"\nDone. Best val MAE: {best_val_mae:.4f}")
    print(f"Saved finetuned_model_best.pt / finetuned_model_final.pt to {out_dir}")
    print("\nNEXT STEP: re-run precompute_embeddings.py using THIS finetuned backbone "
          "to refresh the cached embeddings for A4/A5, since the backbone weights changed.")


if __name__ == "__main__":
    main()