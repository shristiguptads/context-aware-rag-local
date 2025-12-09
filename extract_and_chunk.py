# extract_and_chunk.py
# Extract text from PDFs in ../docs, chunk into passages, save passages.json

import os, json
import fitz  # PyMuPDF
from tqdm import tqdm
import re

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "passages.json")
CHUNK_SIZE = 500  # chars
CHUNK_OVERLAP = 100

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = []
    for page in doc:
        t = page.get_text("text")
        if t:
            text.append(t)
    return "\n".join(text)

def clean_text(t):
    t = t.replace("\r", "\n")
    t = re.sub(r'\n{2,}', '\n\n', t)
    return t.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return chunks

def main():
    all_passages = []
    for fname in os.listdir(DOCS_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(DOCS_DIR, fname)
        print("Reading", fname)
        text = extract_text_from_pdf(path)
        text = clean_text(text)
        if len(text) < 50:
            continue
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            all_passages.append({
                "id": f"{fname}__{i}",
                "source": fname,
                "text": c
            })

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(all_passages, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_passages)} passages to {OUTPUT}")

if __name__ == "__main__":
    main()
