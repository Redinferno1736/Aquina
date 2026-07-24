# scripts/download_atcoder.py
import requests
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_DIR

def download():
    url = "https://kenkoooo.com/atcoder/resources/problems.json"
    headers = {"User-Agent": "ASTEfficiencyChecker"}
    response = requests.get(url, headers=headers)
    data = response.json()
    
    out_file = os.path.join(RAW_DIR, "atcoder_raw.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved AtCoder problems to {out_file}")

if __name__ == "__main__":
    download()