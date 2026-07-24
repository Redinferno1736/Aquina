# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

EMBEDDINGS_FILE = os.path.join(PROCESSED_DIR, "problem_embeddings.npy")
DATASET_FILE = os.path.join(PROCESSED_DIR, "normalized_problems.jsonl")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Ensure directories exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)