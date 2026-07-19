from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("clinc_oos", "small")

# Get the mapping from integer ID -> intent name
intent_names = dataset["train"].features["intent"].names

print("Total intents:", len(intent_names))
print()

print("Intent Mapping:")
for i, name in enumerate(intent_names):
    print(f"{i:3} -> {name}")

# Convert train split to pandas DataFrame
train_df = dataset["train"].to_pandas()

# Replace integer IDs with intent names
train_df["intent"] = train_df["intent"].apply(lambda x: intent_names[x])

# Save readable CSV
train_df.to_csv("train_readable.csv", index=False)

print("\nCreated train_readable.csv")
print("\nFirst 10 rows:")
print(train_df.head(10))