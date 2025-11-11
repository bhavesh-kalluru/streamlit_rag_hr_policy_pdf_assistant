import os
import faiss
import pickle
import numpy as np
from utils.embedding import get_embedding

def load_index(index_dir="index"):
    index = faiss.read_index(os.path.join(index_dir, "chunks.index"))
    with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
        data = pickle.load(f)
    return index, data["texts"], data["metadatas"]

def retrieve(query: str, k: int = 4, index_dir="index"):
    index, texts, metas = load_index(index_dir=index_dir)
    q = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "text": texts[idx],
            "metadata": metas[idx]
        })
    return results
