"""
Aquina A3 - Embedding precomputation
Since the backbone is frozen, its output for a given input never changes
across epochs. Instead of re-running CodeBERT on every sample every epoch
(which is the actual bottleneck), run it ONCE per sample here and cache the
pooled embeddings + labels to disk. Training then becomes a tiny MLP over
cached vectors - seconds per epoch instead of hours.

Run:
    python precompute_embeddings.py --data ./data/labeled_functions.parquet --out ./data/embeddings.pt

This uses fp16 + a larger batch size since we only need forward passes
(no gradients, no backward pass) - much lighter on a 4GB GPU than training.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

BACKBONE_NAME = "microsoft/codebert-base"
MAX_TOKEN_LEN = 512


class RawCodeDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.codes = df["code"].tolist()
        self.lang_ids = df["language_id"].tolist()
        self.cc = df["cyclomatic_complexity"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.codes[idx],
            truncation=True,
            max_length=MAX_TOKEN_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "language_id": self.lang_ids[idx],
            "cc": self.cc[idx],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="./data/embeddings.pt")
    parser.add_argument("--batch_size", type=int, default=32,
                         help="Can go higher than training batch size - no gradients stored")
    parser.add_argument("--max_token_len", type=int, default=MAX_TOKEN_LEN)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] Using device: {device}")
    if device.type == "cuda":
        print(f"[setup] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[setup] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"[setup] Loading data from {args.data} ...")
    df = pd.read_parquet(args.data)
    print(f"[setup] Loaded {len(df)} rows")

    print(f"[setup] Loading tokenizer + backbone {BACKBONE_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    backbone = AutoModel.from_pretrained(BACKBONE_NAME).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print("[setup] Backbone loaded, frozen, in eval mode")

    ds = RawCodeDataset(df, tokenizer)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"[setup] {len(loader)} batches at batch_size={args.batch_size}")

    all_embeddings = []
    all_lang_ids = []
    all_cc = []

    use_amp = device.type == "cuda"
    print(f"[setup] Mixed precision (fp16): {use_amp}")
    print("=" * 60)
    print("Computing embeddings (one-time cost) ...")
    print("=" * 60)

    with torch.no_grad():
        for batch in tqdm(loader, desc="[embed]", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
                token_embeds = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).float()
                summed = (token_embeds * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-6)
                pooled = summed / counts  # (batch, 768)

            all_embeddings.append(pooled.float().cpu())
            all_lang_ids.extend(batch["language_id"].tolist())
            all_cc.extend(batch["cc"].tolist())

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    lang_ids_tensor = torch.tensor(all_lang_ids, dtype=torch.long)
    cc_tensor = torch.tensor(all_cc, dtype=torch.float32)

    print(f"[done] embeddings shape: {embeddings_tensor.shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": embeddings_tensor,
        "language_id": lang_ids_tensor,
        "cyclomatic_complexity": cc_tensor,
    }, out_path)
    print(f"[done] Saved cached embeddings to {out_path}")
    print(f"[done] File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()