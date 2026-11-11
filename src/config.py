import os
from pathlib import Path
from dotenv import load_dotenv

# Project root is one level above src/
ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"

# Explicitly load .env from root folder
load_dotenv(dotenv_path=ENV_PATH, override=True)

DATA_DIR = ROOT_DIR / "data" / "processed"
CHUNKS_PATH = DATA_DIR / "chunks.json"
FAISS_PATH = DATA_DIR / "faiss_index.bin"
BM25_PATH = DATA_DIR / "bm25_index.pkl"

BI_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"