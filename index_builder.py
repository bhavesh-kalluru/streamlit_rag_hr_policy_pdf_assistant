import os
import faiss
import pickle
from typing import List, Dict, Tuple
from utils.embedding import get_embedding
from pathlib import Path

CHUNK_SIZE = 500    # words per chunk (approximate)
CHUNK_OVERLAP = 100

def read_txt_files(data_dir: str) -> Dict[str, str]:
    texts = {}
    for p in Path(data_dir).glob("**/*.txt"):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if content:
                texts[str(p)] = content
    return texts

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        step = max(1, chunk_size - overlap)
        i += step
    return chunks

def build_faiss_index(chunks: List[str]):
    vectors = [get_embedding(ch) for ch in chunks]
    import numpy as np
    mat = np.array(vectors).astype("float32")
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index

def run(data_dir="data", index_dir="index"):
    os.makedirs(index_dir, exist_ok=True)
    texts = read_txt_files(data_dir)
    if not texts:
        raise RuntimeError(f"No .txt files with content found in {data_dir}. Upload PDFs/TXTs first.")
    doc_chunks = []
    metadatas = []
    for path, content in texts.items():
        chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, ch in enumerate(chunks):
            doc_chunks.append(ch)
            metadatas.append({"source": path, "chunk_id": i})
    if not doc_chunks:
        raise RuntimeError("No chunks created—check chunk size/overlap or input documents.")
    index = build_faiss_index(doc_chunks)
    faiss.write_index(index, os.path.join(index_dir, "chunks.index"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump({"texts": doc_chunks, "metadatas": metadatas}, f)
    print(f"Indexed {len(doc_chunks)} chunks from {len(texts)} files.")

if __name__ == "__main__":
    run()
