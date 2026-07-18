from pathlib import Path
import pandas as pd

# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ----------------------------
# Read dataset
# ----------------------------
df = pd.read_csv(DATA_DIR / "raw_videos.csv")

print(f"Original rows: {len(df)}")

# ----------------------------
# Clean data
# ----------------------------

# Remove rows with missing titles
df = df.dropna(subset=["title"])

# Fill missing descriptions
df["description"] = df["description"].fillna("").astype(str).str.strip()

# Clean title and channel
df["title"] = df["title"].astype(str).str.strip()
df["channel"] = df["channel"].astype(str).str.strip()

# Remove duplicate videos
df = df.drop_duplicates(subset=["video_id"])

print(f"Rows after cleaning: {len(df)}")

# ----------------------------
# Build model input
# ----------------------------

df["text"] = (
    "[TITLE] " + df["title"] +
    " [CHANNEL] " + df["channel"] +
    " [DESC] " + df["description"].str[:250]
)

# ----------------------------
# Channel Split
# ----------------------------

val_channels = [
    "CodeAesthetic",
    "Computerphile",
    "MIT OpenCourseWare",
    "Geeky Shows",
    "Netflix",
    "Round2hell",
    "Sony Music India",
    "Dude Perfect"
]

test_channels = [
    "Programming with Mosh",
    "Stanford Online",
    "CodeWithHarry",
    "Apna College",
    "MrBeast",
    "CarryMinati",
    "IGN",
    "Sidemen"
]

val_df = df[df["channel"].isin(val_channels)]

test_df = df[df["channel"].isin(test_channels)]

train_df = df[
    ~df["channel"].isin(val_channels + test_channels)
]

# ----------------------------
# Save
# ----------------------------

train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

# ----------------------------
# Summary
# ----------------------------

print("\n=========================")
print("Dataset Split Complete")
print("=========================")

print(f"Train : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test : {len(test_df)}")

print("\nTrain Labels")
print(train_df["label"].value_counts())

print("\nValidation Labels")
print(val_df["label"].value_counts())

print("\nTest Labels")
print(test_df["label"].value_counts())