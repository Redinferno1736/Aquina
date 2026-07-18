"""
train.py

Trains a DistilBERT-based binary classifier for the Aquina YouTube
Classifier module.

Labels:
    0 = Entertainment
    1 = Coding

Input:
    data/train.csv
    data/val.csv

    Required columns: "text", "label"

Output:
    models/youtube_classifier/  (best model, by validation accuracy)

Run:
    python scripts/train.py
"""

import time
import numpy as np
import pandas as pd
import evaluate
from pathlib import Path
from datetime import timedelta

from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"

OUTPUT_DIR = MODELS_DIR / "youtube_classifier"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
NUM_LABELS = 2

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------

def load_datasets():
    """
    Loads train.csv and val.csv from the data directory, validates
    that the required columns exist, and converts them into
    HuggingFace Dataset objects.
    """
    print("=" * 60)
    print("STEP 1/5: Loading dataset...")
    print("=" * 60)

    print(f"  Looking for train.csv at: {TRAIN_CSV}")
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Could not find {TRAIN_CSV}")
    print("  Found train.csv")

    print(f"  Looking for val.csv at:   {VAL_CSV}")
    if not VAL_CSV.exists():
        raise FileNotFoundError(f"Could not find {VAL_CSV}")
    print("  Found val.csv")

    print("  Reading train.csv into memory...")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"  Read {len(train_df)} raw rows from train.csv")

    print("  Reading val.csv into memory...")
    val_df = pd.read_csv(VAL_CSV)
    print(f"  Read {len(val_df)} raw rows from val.csv")

    required_cols = {"text", "label"}
    for name, df in [("train.csv", train_df), ("val.csv", val_df)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
    print("  Column check passed: 'text' and 'label' present in both files")

    # Only keep the columns the model actually needs.
    print("  Dropping unused columns and rows with missing values...")
    before_train, before_val = len(train_df), len(val_df)
    train_df = train_df[["text", "label"]].dropna()
    val_df = val_df[["text", "label"]].dropna()
    print(f"  Train: {before_train} -> {len(train_df)} rows after cleaning")
    print(f"  Val:   {before_val} -> {len(val_df)} rows after cleaning")

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)

    print(f"  Final train rows: {len(train_df)}")
    print(f"  Final val rows:   {len(val_df)}")
    print("  Train label distribution:")
    print(f"    {train_df['label'].value_counts().to_dict()}")
    print("  Val label distribution:")
    print(f"    {val_df['label'].value_counts().to_dict()}")

    print("  Converting pandas DataFrames to HuggingFace Datasets...")
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))
    print("  Dataset objects created.")
    print("STEP 1/5 complete.\n")

    return train_dataset, val_dataset


# -------------------------------------------------------------------
# Tokenization
# -------------------------------------------------------------------

def tokenize_datasets(train_dataset, val_dataset, tokenizer):
    """
    Tokenizes the "text" column for both datasets, truncating to
    MAX_LENGTH tokens. Padding is left to the data collator
    (dynamic padding per batch), not applied here.
    """
    print("=" * 60)
    print("STEP 2/5: Tokenizing...")
    print("=" * 60)
    print(f"  Tokenizer: {MODEL_NAME}")
    print(f"  Max sequence length: {MAX_LENGTH} tokens")

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    t0 = time.time()
    print("  Tokenizing train set...")
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in train_dataset.column_names if c not in ("label",)],
    )
    print(f"  Train set tokenized in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("  Tokenizing val set...")
    val_dataset = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in val_dataset.column_names if c not in ("label",)],
    )
    print(f"  Val set tokenized in {time.time() - t0:.1f}s")

    # Trainer expects the label column to be named "labels".
    print("  Renaming 'label' -> 'labels' for Trainer compatibility...")
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")

    print("  Setting torch tensor format...")
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    print("STEP 2/5 complete.\n")

    return train_dataset, val_dataset


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and f1 (binary average)
    from the Trainer's evaluation predictions.
    """
    print("  Running evaluation on validation set...")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )

    print("  " + "-" * 40)
    print(f"  Validation Accuracy:  {accuracy:.4f}")
    print(f"  Validation Precision: {precision:.4f}")
    print(f"  Validation Recall:    {recall:.4f}")
    print(f"  Validation F1:        {f1:.4f}")
    print("  " + "-" * 40)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

def build_trainer(train_dataset, val_dataset, tokenizer, model):
    """
    Builds the HuggingFace Trainer with CPU-friendly training
    arguments and dynamic padding via DataCollatorWithPadding.
    """
    print("=" * 60)
    print("STEP 4/5: Building Trainer...")
    print("=" * 60)
    print(f"  Batch size (per device):        {BATCH_SIZE}")
    print(f"  Gradient accumulation steps:    {GRAD_ACCUM_STEPS}")
    print(f"  Effective batch size:           {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"  Epochs:                         {NUM_EPOCHS}")
    print(f"  Learning rate:                  {LEARNING_RATE}")
    print(f"  Weight decay:                   {WEIGHT_DECAY}")
    print(f"  Device:                         CPU (no_cuda=True)")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=False,
        no_cuda=True,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("  Trainer built successfully.")
    print("STEP 4/5 complete.\n")

    return trainer


def train_model():
    """
    Orchestrates the full training pipeline: load data, tokenize,
    build model + trainer, train, and save the best model.
    """
    overall_start = time.time()

    print("#" * 60)
    print("# AQUINA YOUTUBE CLASSIFIER - TRAINING PIPELINE")
    print("#" * 60)
    print()

    print("Creating output directories...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Models dir:     {MODELS_DIR}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Checkpoint dir: {CHECKPOINT_DIR}\n")

    train_dataset, val_dataset = load_datasets()

    print("=" * 60)
    print("STEP 2/5 (continued): Loading tokenizer...")
    print("=" * 60)
    print(f"  Loading tokenizer for: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("  Tokenizer loaded.\n")

    train_dataset, val_dataset = tokenize_datasets(train_dataset, val_dataset, tokenizer)

    print("=" * 60)
    print("STEP 3/5: Loading pretrained model...")
    print("=" * 60)
    print(f"  Loading {MODEL_NAME} with a {NUM_LABELS}-label classification head...")
    model_load_start = time.time()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    print(f"  Model loaded in {time.time() - model_load_start:.1f}s")
    print("STEP 3/5 complete.\n")

    trainer = build_trainer(train_dataset, val_dataset, tokenizer, model)

    print("=" * 60)
    print("STEP 5/5: Training...")
    print("=" * 60)
    print(f"  Starting training across {NUM_EPOCHS} epochs.")
    print("  Progress and loss will print every "
          f"{LOGGING_STEPS} steps, and full metrics after each epoch.\n")

    train_start = time.time()
    trainer.train()
    train_elapsed = time.time() - train_start
    print(f"\n  Training loop finished in {timedelta(seconds=int(train_elapsed))} "
          f"({train_elapsed:.1f}s)")

    print("\n  Saving best model (selected by validation accuracy)...")
    save_start = time.time()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"  Model + tokenizer saved in {time.time() - save_start:.1f}s")
    print("STEP 5/5 complete.\n")

    overall_elapsed = time.time() - overall_start

    print("#" * 60)
    print("# TRAINING COMPLETE")
    print("#" * 60)
    print(f"  Best model saved to: {OUTPUT_DIR}")
    print(f"  Total pipeline time: {timedelta(seconds=int(overall_elapsed))} "
          f"({overall_elapsed:.1f} seconds)")
    print(f"  Of which, model training took: "
          f"{timedelta(seconds=int(train_elapsed))} ({train_elapsed:.1f} seconds)")
    print("#" * 60)


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    train_model()