# engine/matcher.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import config

class ProblemMatcher:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.embeddings = None
        self.dataset = []
        self._load_data()

    def _load_data(self):
        if not os.path.exists(config.EMBEDDINGS_FILE) or not os.path.exists(config.DATASET_FILE):
            return

        self.embeddings = np.load(config.EMBEDDINGS_FILE)
        with open(config.DATASET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                self.dataset.append(json.loads(line.strip()))

    def match(self, code: str, detected_algorithm: str):
        if self.embeddings is None or len(self.dataset) == 0:
            return None, 0.0

        query = f"Code solving problem using {detected_algorithm}. Code: {code[:200]}"
        query_embedding = self.model.encode([query])[0]

        norm_q = np.linalg.norm(query_embedding)
        norm_db = np.linalg.norm(self.embeddings, axis=1)
        similarities = np.dot(self.embeddings, query_embedding) / (norm_db * norm_q)

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > 0.4:
            return self.dataset[best_idx], float(best_score)
        return None, 0.0