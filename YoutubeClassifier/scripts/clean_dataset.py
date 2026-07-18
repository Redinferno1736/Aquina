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

print("Original rows:", len(df))

# ----------------------------
# Remove wrongly matched channel
# ----------------------------

wrong_channels = [
    "The PrimeTime"
     "CodeAesthetic"
]

df = df[~df["channel"].isin(wrong_channels)]

print("Rows after removing wrong channels:", len(df))

# ----------------------------
# Save cleaned dataset
# ----------------------------

df.to_csv(DATA_DIR / "raw_videos.csv", index=False)

print("\nDataset cleaned successfully!")