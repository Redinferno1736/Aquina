import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

# ==========================
# Paths
# ==========================

MODEL_DIR = "../models/youtube_classifier"
TEST_FILE = "../data/unseen_test_dataset.csv"

# ==========================
# Load Model
# ==========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(TEST_FILE)

dataset = Dataset.from_pandas(df)

# ==========================
# Tokenization
# ==========================

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,
        max_length=256,
    )

dataset = dataset.map(tokenize, batched=True)

# ==========================
# Trainer
# ==========================

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ==========================
# Predict
# ==========================

predictions = trainer.predict(dataset)

preds = torch.argmax(
    torch.tensor(predictions.predictions),
    dim=1
).numpy()

labels = df["label"].values

# ==========================
# Metrics
# ==========================

accuracy = accuracy_score(labels, preds)

precision, recall, f1, _ = precision_recall_fscore_support(
    labels,
    preds,
    average="binary"
)

print("\n========== RESULTS ==========\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix\n")
print(confusion_matrix(labels, preds))

print("\nClassification Report\n")
print(classification_report(
    labels,
    preds,
    target_names=["Entertainment", "Coding"]
))

# ==========================
# Save Predictions
# ==========================

df["prediction"] = preds
df["predicted_label"] = df["prediction"].map({
    0: "Entertainment",
    1: "Coding"
})

df["actual_label"] = df["label"].map({
    0: "Entertainment",
    1: "Coding"
})

df["correct"] = df["label"] == df["prediction"]

df.to_csv("../data/unseen_predictions.csv", index=False)

print("\nPredictions saved to:")
print("data/unseen_predictions.csv")