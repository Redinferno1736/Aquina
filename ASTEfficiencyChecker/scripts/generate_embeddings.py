# scripts/generate_embeddings.py
import json
import os
import numpy as np
import sys
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def generate():
    if not os.path.exists(config.DATASET_FILE):
        print("Normalized dataset not found. Run build_dataset.py first.")
        return

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    texts = []
    
    with open(config.DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            texts.append(f"{item['title']} - {item['problem_statement']} Tags: {','.join(item['tags'])}")
    
    if not texts:
        print("No texts to embed.")
        return

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    np.save(config.EMBEDDINGS_FILE, embeddings)
    print(f"Embeddings saved to {config.EMBEDDINGS_FILE}")

if __name__ == "__main__":
    generate()