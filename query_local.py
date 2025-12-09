# query_local.py
# Simple local query script: embed query, search FAISS, print top-k contexts.
# Optionally generate a short answer with a transformer summarizer (if available).

import os, json, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

BASE = os.path.join(os.path.dirname(__file__), "..")
INDEX_FILE = os.path.join(BASE, "faiss.index")
META_FILE = os.path.join(BASE, "meta.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3

def load():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, meta, model

def generate_answer_with_model(context, question):
    # Try a small summarization model -- fallback if not installed or out of RAM
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nProvide a concise answer based only on the context."
        out = summarizer(prompt, max_length=150, min_length=20, truncation=True)
        return out[0]["summary_text"]
    except Exception as e:
        # fallback: return the context directly
        return ("[Top context passages]\n" + context[:1500] + "\n\n(If you want a generated answer, install transformers & a summarization model.)")

def query_loop():
    index, meta, embedder = load()
    ids = meta["ids"]
    passages = meta["passages"]

    print("Ready. Type your question (or 'exit').")
    while True:
        q = input("\n> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        q_vec = embedder.encode([q], convert_to_numpy=True)
        D, I = index.search(q_vec, TOP_K)
        hits = []
        context_texts = []
        for idx in I[0]:
            if idx < 0: 
                continue
            p = passages[idx]
            hits.append(p)
            context_texts.append(p["text"])
        context = "\n\n---\n\n".join(context_texts)
        answer = generate_answer_with_model(context, q)
        print("\n==== RAG Answer (local) ====\n")
        print(answer)
        print("\n==== SOURCES ====")
        for h in hits:
            print("-", h["id"])

if __name__ == "__main__":
    query_loop()
