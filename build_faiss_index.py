# build_faiss_index.py
# Reads passages.json, computes embeddings (sentence-transformers), builds FAISS index and saves vectors+metadata.

import os, json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import pickle

BASE = os.path.join(os.path.dirname(__file__), "..")
PASSAGES_FILE = os.path.join(BASE, "passages.json")
INDEX_FILE = os.path.join(BASE, "faiss.index")
META_FILE = os.path.join(BASE, "meta.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"  # small and fast

def main():
    with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
        passages = json.load(f)

    texts = [p["text"] for p in passages]
    ids = [p["id"] for p in passages]

    print("Loading embedder...")
    model = SentenceTransformer(MODEL_NAME)

    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    print("Embedding dimension:", dim)

    # build FAISS index
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"ids": ids, "passages": passages}, f)

    print("Saved FAISS index to", INDEX_FILE)
    print("Saved metadata to", META_FILE)

if __name__ == "__main__":
    main()

