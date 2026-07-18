import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("YOUTUBE_API_KEY")

print("API Key loaded:", API_KEY[:10] + "..." if API_KEY else "NOT FOUND")

youtube = build("youtube", "v3", developerKey=API_KEY)

# Read channel list
channels = pd.read_csv("data/channels.csv")

results = []

for _, row in channels.iterrows():
    channel_name = row["channel"]
    label = row["label"]

    try:
        # Search for the channel
        search_request = youtube.search().list(
            q=channel_name,
            part="snippet",
            type="channel",
            maxResults=1
        )

        search_response = search_request.execute()

        if len(search_response["items"]) == 0:
            print(f"❌ Channel not found: {channel_name}")
            continue

        channel_id = search_response["items"][0]["snippet"]["channelId"]

        # Get uploads playlist
        channel_request = youtube.channels().list(
            id=channel_id,
            part="snippet,contentDetails"
        )

        channel_response = channel_request.execute()

        actual_name = channel_response["items"][0]["snippet"]["title"]

        uploads_playlist = (
            channel_response["items"][0]
            ["contentDetails"]
            ["relatedPlaylists"]
            ["uploads"]
        )

        results.append({
            "channel": actual_name,
            "channel_id": channel_id,
            "uploads_playlist": uploads_playlist,
            "label": label
        })

        print(f"✅ {actual_name}")

    except Exception as e:
        print(f"❌ Error for {channel_name}: {e}")

# Save results
df = pd.DataFrame(results)
df.to_csv("data/upload_playlists.csv", index=False)

print("\n🎉 Finished!")
print(f"Total channels found: {len(df)}")
print("Saved to: data/upload_playlists.csv")