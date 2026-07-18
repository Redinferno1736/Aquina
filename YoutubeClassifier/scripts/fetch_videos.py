import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# -------------------------------
# Load API Key
# -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)

DATA_DIR = PROJECT_ROOT / "data"

playlists = pd.read_csv(DATA_DIR / "upload_playlists.csv")

MAX_VIDEOS_PER_CHANNEL = 200

all_videos = []

# -------------------------------
# Loop through every channel
# -------------------------------

for idx, row in playlists.iterrows():

    channel = row["channel"]
    playlist_id = row["uploads_playlist"]
    label = row["label"]

    print(f"\n[{idx+1}/{len(playlists)}] {channel}")

    collected = 0
    next_page = None

    while True:

        try:

            request = youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=50,
                pageToken=next_page
            )

            response = request.execute()

            for item in response["items"]:

                snippet = item["snippet"]

                all_videos.append({
                    "video_id": snippet["resourceId"]["videoId"],
                    "title": snippet["title"],
                    "description": snippet["description"],
                    "channel": snippet["channelTitle"],
                    "published_at": snippet["publishedAt"],
                    "label": label
                })

                collected += 1

                if collected >= MAX_VIDEOS_PER_CHANNEL:
                    break

            print(f"Collected {collected} videos", end="\r")

            if collected >= MAX_VIDEOS_PER_CHANNEL:
                break

            next_page = response.get("nextPageToken")

            if not next_page:
                break

            time.sleep(0.1)

        except HttpError as e:

            print(e)
            time.sleep(5)

            continue

    print(f"✓ {channel} : {collected} videos")

# -------------------------------
# Save CSV
# -------------------------------

df = pd.DataFrame(all_videos)

df.to_csv(DATA_DIR / "raw_videos.csv", index=False)

print("\n===================================")
print("Dataset Created Successfully")
print(f"Total Videos : {len(df)}")
print("Saved to data/raw_videos.csv")
print("===================================")