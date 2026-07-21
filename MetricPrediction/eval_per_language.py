"""
Aquina A3 - Per-language evaluation
Checks whether the shared regression head performs consistently across all
6 languages, or whether one language is quietly worse than the others
(the empirical check we said we'd do instead of assuming shared-head works
equally well everywhere).

Run:
    python eval_per_language.py --embeddings ./data/embeddings.pt --checkpoint ./models/head_best.pt
"""

import argparse
import math

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


def compute_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    mae = np.mean(np.abs(preds - targets))
    rmse = math.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": len(preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
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
    # Same seed as train_head.py so this is the SAME val split it was evaluated on
    _, val_ds = random_split(dataset, [train_len, val_len],
                              generator=torch.Generator().manual_seed(42))
    print(f"[setup] Val set size: {len(val_ds)} (same split used during training)")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RegressionHead().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"[setup] Loaded checkpoint from {args.checkpoint}")

    preds_by_lang = {i: [] for i in range(NUM_LANGUAGES)}
    targets_by_lang = {i: [] for i in range(NUM_LANGUAGES)}

    with torch.no_grad():
        for emb, lang, target in val_loader:
            emb, lang = emb.to(device), lang.to(device)
            pred = model(emb, lang).cpu().squeeze(1)
            for p, t, l in zip(pred.tolist(), target.tolist(), lang.cpu().tolist()):
                preds_by_lang[l].append(p)
                targets_by_lang[l].append(t)

    print("\n" + "=" * 70)
    print(f"{'Language':<12} {'N':>8} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("=" * 70)
    overall_preds, overall_targets = [], []
    for lang_id in range(NUM_LANGUAGES):
        p, t = preds_by_lang[lang_id], targets_by_lang[lang_id]
        if not p:
            continue
        m = compute_metrics(p, t)
        print(f"{LANG_ID_TO_NAME[lang_id]:<12} {m['n']:>8} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")
        overall_preds.extend(p)
        overall_targets.extend(t)

    overall = compute_metrics(overall_preds, overall_targets)
    print("-" * 70)
    print(f"{'OVERALL':<12} {overall['n']:>8} {overall['mae']:>8.4f} {overall['rmse']:>8.4f} {overall['r2']:>8.4f}")
    print("=" * 70)
    print("\nIf one language's MAE/R2 is noticeably worse than the others, that's")
    print("evidence the shared head may need per-language calibration - otherwise,")
    print("the shared-head approach is validated.")


if __name__ == "__main__":
    main()