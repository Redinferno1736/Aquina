from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorWithPadding
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models" / "youtube_classifier"
TEST_CSV = PROJECT_ROOT / "data" / "test.csv"

# --------------------------------------------------
# Load model
# --------------------------------------------------

print("=" * 60)
print("Loading trained model...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

print("Model loaded.\n")

# --------------------------------------------------
# Load test set
# --------------------------------------------------

print("=" * 60)
print("Loading test dataset...")
print("=" * 60)

test_df = pd.read_csv(TEST_CSV)

test_df = test_df[["text", "label"]].dropna()

test_df["label"] = test_df["label"].astype(int)

print(f"Total test samples: {len(test_df)}")

dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

# --------------------------------------------------
# Tokenization
# --------------------------------------------------

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
    )

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.rename_column("label", "labels")

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("\nRunning predictions...\n")

predictions = trainer.predict(dataset)

preds = predictions.predictions.argmax(axis=1)

labels = predictions.label_ids

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(labels, preds)

precision, recall, f1, _ = precision_recall_fscore_support(
    labels,
    preds,
    average="binary",
)

print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix")

print(confusion_matrix(labels, preds))

print("\nClassification Report")

print(classification_report(labels, preds))

# --------------------------------------------------
# Save predictions
# --------------------------------------------------

test_df["prediction"] = preds

test_df["correct"] = (
    test_df["label"] == test_df["prediction"]
)

OUTPUT = PROJECT_ROOT / "data" / "predictions.csv"

test_df.to_csv(OUTPUT, index=False)

print(f"\nPredictions saved to:\n{OUTPUT}")