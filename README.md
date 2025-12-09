# Context-Aware RAG Local

## Overview
This project implements a **Context-Aware Retrieval-Augmented Generation (RAG) system** locally.  
It allows users to extract and chunk textual data, build a FAISS index for semantic search, and generate context-aware responses using sentence embeddings.

---

## Features
- Extract text passages from documents (PDF, DOCX, TXT, etc.)
- Chunk text into manageable passages for efficient retrieval
- Build FAISS index for fast semantic search
- Integrate with `sentence-transformers` for embedding generation
- Ready-to-run scripts for local experimentation

---

## Project Structure
context-aware-rag-local/
├─ scripts/

│ ├─ extract_and_chunk.py # Extracts and chunks passages

│ ├─ build_faiss_index.py # Builds FAISS index

├─ passages.json # Stores extracted passages



├─ README.md

└─ requirements.txt # Python dependencies

---

## Setup Instructions

### 1. Clone or Download
```bash
git clone <your-repo-url>
cd context-aware-rag-local
### 2. Create Virtual Environment
python -m venv venv
.\venv\Scripts\activate       # PowerShell
# OR
venv\Scripts\activate.bat     # Command Prompt
###3. Upgrade pip
python -m pip install --upgrade pip
###4. Install Dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers faiss-cpu pymupdf tqdm flask transformers

Usage
Extract and Chunk Passages
python scripts/extract_and_chunk.py

---

Saves passages to passages.json.

Build FAISS Index
python scripts/build_faiss_index.py
Builds a FAISS index for semantic search.

---

Notes

Avoid spaces in the project folder path to prevent DLL or path issues.

For Windows users, if you encounter WinError 1114, use the Force CPU Mode step above.

Always use the virtual environment for Python commands to avoid conflicts.

License

MIT License



---
