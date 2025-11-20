import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"   # or another embedding model you’ve pulled

def get_embedding(text: str):
    """Call Ollama embeddings API and return a numpy vector."""
    payload = {
        "model": MODEL_NAME,
        "prompt": text,
    }
    resp = requests.post(OLLAMA_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return np.array(data["embedding"], dtype=np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    # small epsilon to avoid division by zero
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b) / denom)

if __name__ == "__main__":
    sentences = [
        "The cat is sleeping on the sofa.",
        "The cat is sitting on the table.",
        "A dog is resting on the couch.",
        "I am driving to the office in my car.",
        "my pc is fast",
        "my apple was chip but works fine",
        "Apples are tasty"
    ]

    # Get embeddings
    embs = [get_embedding(s) for s in sentences]

    # Compute pairwise similarities
    n = len(sentences)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(embs[i], embs[j])
            print(f"Similarity({i}, {j})")
            print(f"  '{sentences[i]}'")
            print(f"  '{sentences[j]}'")
            print(f"  -> {sim:.4f}\n")