from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

df = pd.read_csv(DATA_DIR / "raw_videos.csv")

print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print(f"\nTotal Rows: {len(df)}")

print("\nLabel Distribution")
print(df["label"].value_counts())

print("\nLabel Percentage")
print((df["label"].value_counts(normalize=True) * 100).round(2))

print("\nVideos Per Channel")
print(df.groupby("channel").size().sort_values(ascending=False))

print("\nDuplicate Video IDs")
print(df["video_id"].duplicated().sum())

print("\nEmpty Descriptions")
empty = (df["description"].fillna("").str.strip() == "").sum()
print(f"{empty} ({empty/len(df)*100:.2f}%)")

print("\nAverage Title Length")
print(df["title"].str.len().mean())

print("\nAverage Description Length")
print(df["description"].fillna("").str.len().mean())

print("\nMissing Values")
print(df.isnull().sum())

print("\nTop 10 Largest Channels")
print(df.groupby("channel").size().sort_values(ascending=False).head(10))

print("\nTop 10 Smallest Channels")
print(df.groupby("channel").size().sort_values().head(10))