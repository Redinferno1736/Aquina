"""
train.py

Production-quality training script for Aquina's Intent Classification model.
Base model: distilbert-base-uncased (DistilBertForSequenceClassification)

This script:
- Loads custom Aquina dataset (train/val/test CSVs)
- Ignores the slot column (this is an intent-classification-only model)
- Automatically infers intent labels from the data (no hardcoding)
- Encodes labels with sklearn's LabelEncoder and saves the encoder
- Tokenizes text with AutoTokenizer
- Trains using HuggingFace Trainer API
- Evaluates on validation and test sets (accuracy, precision, recall, F1)
- Saves model, tokenizer, and label encoder to models/
- Packages a lightweight, version-safe inference wrapper into a single
  .pkl file (aquina_intent_pipeline.pkl) for easy integration into the
  Aquina pipeline
- Prints status/progress messages with elapsed and estimated times so the
  terminal never looks "stuck" during long-running phases.
"""

import os
import time
import random
import pickle

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
    set_seed,
)

# ------------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
OUTPUT_DIR = "models"

TRAIN_CSV = "aquina_train.csv"
VAL_CSV = "aquina_val.csv"
TEST_CSV = "aquina_test.csv"


# ------------------------------------------------------------------------
# Small helper for readable elapsed-time strings (e.g. "1m 23s")
# ------------------------------------------------------------------------
def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {secs}s"
    if mins > 0:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def stage_start(msg: str) -> float:
    """Print a stage-start message and return the start timestamp."""
    print(f"\n[STAGE START] {msg} ...")
    return time.time()


def stage_end(msg: str, start_time: float):
    """Print a stage-end message with elapsed time."""
    elapsed = time.time() - start_time
    print(f"[STAGE DONE ] {msg} — took {format_seconds(elapsed)}")


# ------------------------------------------------------------------------
# Trainer callback: prints a clear per-epoch status line with ETA
# ------------------------------------------------------------------------
class ProgressCallback(TrainerCallback):
    """
    Custom callback to print clear, human-readable status updates during
    training so the terminal doesn't look idle. Prints:
      - When training starts (total epochs/steps)
      - After each logging step: loss + elapsed + estimated time remaining
      - After each epoch: a summary line
      - When evaluation runs
    """

    def __init__(self):
        self.train_start_time = None
        self.total_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        self.total_steps = state.max_steps
        print(
            f"\n[TRAINING START] epochs={args.num_train_epochs}, "
            f"total_steps={self.total_steps}, "
            f"batch_size={args.per_device_train_batch_size}, "
            f"device={'cuda' if torch.cuda.is_available() else 'cpu'}"
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Only print during actual training steps (loss present)
        if "loss" in logs:
            elapsed = time.time() - self.train_start_time
            step = state.global_step
            if step > 0 and self.total_steps:
                pct = step / self.total_steps
                est_total = elapsed / pct if pct > 0 else 0
                est_remaining = max(est_total - elapsed, 0)
                print(
                    f"  [step {step}/{self.total_steps} | {pct*100:5.1f}%] "
                    f"loss={logs['loss']:.4f} | "
                    f"elapsed={format_seconds(elapsed)} | "
                    f"ETA={format_seconds(est_remaining)}"
                )

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start_time
        print(
            f"[EPOCH {int(state.epoch)}/{int(args.num_train_epochs)} COMPLETE] "
            f"elapsed so far: {format_seconds(elapsed)}"
        )

    def on_evaluate(self, args, state, control, **kwargs):
        print("  [EVAL] Running evaluation on validation set...")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start_time
        print(f"[TRAINING END] total training time: {format_seconds(elapsed)}")


# ------------------------------------------------------------------------
# Inference wrapper class — this is what gets pickled for pipeline use.
#
# We do NOT pickle the raw torch model/tokenizer objects directly, since
# that is fragile across library versions and machines. Instead we pickle
# a small wrapper that stores the path to the saved model directory and
# the label encoder, then lazily loads the actual model/tokenizer the
# first time .predict() is called. This makes the .pkl file small, and
# lets it survive across transformers/torch version bumps as long as the
# model directory itself is shipped alongside it.
# ------------------------------------------------------------------------
class AquinaIntentPredictor:
    """
    Self-contained intent prediction wrapper for Aquina.

    Usage after unpickling:
        predictor = pickle.load(open("aquina_intent_pipeline.pkl", "rb"))
        intent, confidence = predictor.predict("open chrome please")
    """

    def __init__(self, model_dir: str, label_encoder: LabelEncoder, max_length: int = 64):
        self.model_dir = model_dir
        self.label_encoder = label_encoder
        self.max_length = max_length

        # These are NOT pickled (see __getstate__ below) — they are
        # rebuilt lazily on first use in the new process/environment.
        self._model = None
        self._tokenizer = None
        self._device = None

    def _lazy_load(self):
        """Load the model and tokenizer from disk if not already loaded."""
        if self._model is None or self._tokenizer is None:
            from transformers import AutoTokenizer, DistilBertForSequenceClassification

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
            self._model.to(self._device)
            self._model.eval()

    def predict(self, text: str):
        """
        Predict the intent for a single text string.
        Returns: (predicted_intent: str, confidence: float)
        """
        self._lazy_load()

        inputs = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            confidence, pred_id = torch.max(probs, dim=-1)

        intent = self.label_encoder.inverse_transform([pred_id.item()])[0]
        return intent, confidence.item()

    def predict_batch(self, texts: list):
        """
        Predict intents for a list of text strings.
        Returns: list of (intent: str, confidence: float) tuples
        """
        self._lazy_load()

        inputs = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            confidences, pred_ids = torch.max(probs, dim=-1)

        intents = self.label_encoder.inverse_transform(pred_ids.cpu().numpy())
        return list(zip(intents, confidences.cpu().numpy().tolist()))

    # --------------------------------------------------------------------
    # Custom pickling behavior: only serialize the lightweight metadata
    # (model_dir path + label_encoder). The actual torch model/tokenizer
    # are excluded from the pickle and reloaded lazily after unpickling.
    # --------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        state["_tokenizer"] = None
        state["_device"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ------------------------------------------------------------------------
# Reproducibility: set all relevant random seeds
# ------------------------------------------------------------------------
def set_all_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HuggingFace's own seed setter (covers Trainer internals)


# ------------------------------------------------------------------------
# Data loading and cleaning
# ------------------------------------------------------------------------
def load_and_clean_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file containing text, intent, slot columns.
    We only use 'text' and 'intent'. The 'slot' column is ignored entirely.
    Handles missing values safely by dropping rows with NaN text/intent
    and stripping whitespace.
    """
    print(f"  Reading {path} ...")
    df = pd.read_csv(path)

    # Ensure required columns exist
    required_cols = {"text", "intent"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(
            f"File {path} must contain at least columns: {required_cols}. "
            f"Found: {list(df.columns)}"
        )

    # Keep only the columns we care about (drop slot completely)
    df = df[["text", "intent"]].copy()

    # Handle missing values safely: drop rows with missing text or intent
    before = len(df)
    df.dropna(subset=["text", "intent"], inplace=True)

    # Strip whitespace, drop empty strings after stripping
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]

    dropped = before - len(df)
    if dropped > 0:
        print(f"    -> Dropped {dropped} invalid/empty rows from {path}")

    # Reset index after filtering
    df.reset_index(drop=True, inplace=True)
    print(f"    -> Loaded {len(df)} usable rows from {path}")

    return df


# ------------------------------------------------------------------------
# Metrics computation for Trainer
# ------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1 (weighted average)
    for multi-class intent classification.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    script_start = time.time()

    # --------------------------------------------------------------------
    # Step 0: Reproducibility
    # --------------------------------------------------------------------
    print("=" * 60)
    print("AQUINA INTENT CLASSIFIER — TRAINING SCRIPT")
    print("=" * 60)
    set_all_seeds(SEED)
    print(f"Random seed set to {SEED} for reproducibility.")

    # --------------------------------------------------------------------
    # Step 1: Detect device (GPU if available, else CPU)
    # --------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"GPU detected: {gpu_name} (device count: {gpu_count})")
        print("Training will run on GPU (CUDA). 🚀")
    else:
        print("No GPU detected — training will run on CPU.")
        print("WARNING: CPU training will be significantly slower.")
    print(f"Using device: {device}")

    # --------------------------------------------------------------------
    # Step 2: Load datasets from CSV, ignoring the slot column
    # --------------------------------------------------------------------
    t0 = stage_start("Loading and cleaning CSV datasets")
    train_df = load_and_clean_csv(TRAIN_CSV)
    val_df = load_and_clean_csv(VAL_CSV)
    test_df = load_and_clean_csv(TEST_CSV)
    stage_end("Loading and cleaning CSV datasets", t0)

    # Shuffle training data (important since CSV rows may be grouped by intent)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print("Training data shuffled.")

    # --------------------------------------------------------------------
    # Step 3: Automatically infer labels from the training data
    # (No hardcoding of intent names anywhere in this script)
    # --------------------------------------------------------------------
    t0 = stage_start("Encoding intent labels")
    label_encoder = LabelEncoder()
    all_intents = pd.concat([train_df["intent"], val_df["intent"], test_df["intent"]])
    label_encoder.fit(all_intents)

    num_labels = len(label_encoder.classes_)
    label_mapping = {
        label: int(idx) for idx, label in enumerate(label_encoder.classes_)
    }

    train_df["label"] = label_encoder.transform(train_df["intent"])
    val_df["label"] = label_encoder.transform(val_df["intent"])
    test_df["label"] = label_encoder.transform(test_df["intent"])
    stage_end("Encoding intent labels", t0)

    # --------------------------------------------------------------------
    # Step 4: Save the label encoder to disk for inference-time decoding
    # --------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    label_encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.pkl")
    with open(label_encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {label_encoder_path}")

    # --------------------------------------------------------------------
    # Step 5: Print dataset / label information
    # --------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Number of labels: {num_labels}")
    print(f"Label mapping: {label_mapping}")
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    print("=" * 60)

    # --------------------------------------------------------------------
    # Step 6: Load tokenizer
    # --------------------------------------------------------------------
    t0 = stage_start(f"Loading tokenizer ({MODEL_NAME})")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    stage_end("Loading tokenizer", t0)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    # --------------------------------------------------------------------
    # Step 7: Convert pandas DataFrames into HuggingFace Dataset objects
    # --------------------------------------------------------------------
    t0 = stage_start("Converting to HuggingFace Datasets and tokenizing")
    train_dataset = Dataset.from_pandas(
        train_df[["text", "label"]], preserve_index=False
    )
    val_dataset = Dataset.from_pandas(
        val_df[["text", "label"]], preserve_index=False
    )
    test_dataset = Dataset.from_pandas(
        test_df[["text", "label"]], preserve_index=False
    )

    print("  Tokenizing train split...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    print("  Tokenizing validation split...")
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    print("  Tokenizing test split...")
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    stage_end("Converting to HuggingFace Datasets and tokenizing", t0)

    # Set format for PyTorch tensors (input_ids, attention_mask, label)
    columns_to_keep = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=columns_to_keep)
    val_dataset.set_format(type="torch", columns=columns_to_keep)
    test_dataset.set_format(type="torch", columns=columns_to_keep)

    # --------------------------------------------------------------------
    # Step 8: Load the base model with the correct number of output labels
    # --------------------------------------------------------------------
    t0 = stage_start(f"Loading base model ({MODEL_NAME})")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    model.to(device)
    stage_end("Loading base model", t0)
    print(f"Model moved to device: {device}")

    # --------------------------------------------------------------------
    # Step 9: Data collator (dynamic padding not strictly needed since we
    # already pad to max_length, but included for robustness/consistency)
    # --------------------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --------------------------------------------------------------------
    # Step 10: Define TrainingArguments with sensible defaults
    # --------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=50,
        eval_strategy="epoch",       # modern replacement for evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=SEED,
        disable_tqdm=False,  # keep the built-in progress bar too, as a backup indicator
    )

    # --------------------------------------------------------------------
    # Step 11: Initialize the Trainer (with our custom progress callback)
    # --------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback()],
    )

    # --------------------------------------------------------------------
    # Step 12: Train the model
    # --------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    t0 = time.time()
    trainer.train()
    stage_end("Full training run", t0)

    # --------------------------------------------------------------------
    # Step 13: Evaluate on validation set
    # --------------------------------------------------------------------
    print("\n" + "=" * 60)
    t0 = stage_start("Evaluating on VALIDATION set")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    stage_end("Evaluating on VALIDATION set", t0)
    print(f"Validation metrics: {val_metrics}")

    # --------------------------------------------------------------------
    # Step 14: Evaluate on test set
    # --------------------------------------------------------------------
    print("\n" + "=" * 60)
    t0 = stage_start("Evaluating on TEST set")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    stage_end("Evaluating on TEST set", t0)
    print(f"Test metrics: {test_metrics}")

    # --------------------------------------------------------------------
    # Step 15: Detailed classification report + confusion matrix on TEST
    # --------------------------------------------------------------------
    t0 = stage_start("Generating classification report and confusion matrix")
    test_predictions = trainer.predict(test_dataset)
    test_preds = np.argmax(test_predictions.predictions, axis=-1)
    test_labels = test_predictions.label_ids

    target_names = list(label_encoder.classes_)

    print("\n" + "=" * 60)
    print("Classification Report (Test Set):")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=target_names,
            zero_division=0,
        )
    )

    print("\n" + "=" * 60)
    print("Confusion Matrix (Test Set):")
    cm = confusion_matrix(test_labels, test_preds)
    print(cm)
    stage_end("Generating classification report and confusion matrix", t0)

    # --------------------------------------------------------------------
    # Step 16: Save model, tokenizer, and label encoder to models/
    # --------------------------------------------------------------------
    t0 = stage_start(f"Saving model, tokenizer, and label encoder to {OUTPUT_DIR}/")
    trainer.save_model(OUTPUT_DIR)       # saves model + config
    tokenizer.save_pretrained(OUTPUT_DIR)  # saves tokenizer files
    # label_encoder.pkl was already saved earlier to OUTPUT_DIR
    stage_end(f"Saving model, tokenizer, and label encoder to {OUTPUT_DIR}/", t0)

    # --------------------------------------------------------------------
    # Step 17: Package everything into a single .pkl for pipeline integration
    # --------------------------------------------------------------------
    t0 = stage_start("Packaging inference pipeline into a single .pkl file")

    predictor = AquinaIntentPredictor(
        model_dir=OUTPUT_DIR,
        label_encoder=label_encoder,
        max_length=MAX_LENGTH,
    )

    pipeline_pkl_path = os.path.join(OUTPUT_DIR, "aquina_intent_pipeline.pkl")
    with open(pipeline_pkl_path, "wb") as f:
        pickle.dump(predictor, f)

    stage_end("Packaging inference pipeline into a single .pkl file", t0)
    print(f"Pipeline object saved to {pipeline_pkl_path}")
    print(
        "NOTE: This .pkl references files inside the 'models/' directory "
        "(config.json, model weights, tokenizer files). Ship the whole "
        "'models/' folder together with this .pkl — the model weights "
        "themselves are NOT embedded inside the pickle."
    )

    # --------------------------------------------------------------------
    # Step 18: Final summary printout
    # --------------------------------------------------------------------
    total_elapsed = time.time() - script_start
    print("\n" + "=" * 60)
    print("Training Complete.")
    print(f"Validation Accuracy: {val_metrics.get('eval_accuracy'):.4f}")
    print(f"Validation F1: {val_metrics.get('eval_f1'):.4f}")
    print(f"Test Accuracy: {test_metrics.get('eval_accuracy'):.4f}")
    print(f"Test F1: {test_metrics.get('eval_f1'):.4f}")
    print(f"Model saved to {OUTPUT_DIR}/")
    print(f"Inference pipeline (.pkl) saved to {pipeline_pkl_path}")
    print(f"Total script runtime: {format_seconds(total_elapsed)}")
    print("=" * 60)


if __name__ == "__main__":
    main()