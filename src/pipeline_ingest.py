import sys
import json
import asyncio
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Resolve root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.crawler import run_crawler
from src.extractors import process_item

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_FILE = PROCESSED_DIR / "processed_chunks.json"


def main():
    print("=== STEP 1: CRAWLING URLS ===")
    crawled_items = asyncio.run(run_crawler())

    print("\n=== STEP 2: EXTRACTING TEXT & TABLES ===")
    documents = []
    for item in crawled_items:
        extracted_text = process_item(item)
        if len(extracted_text.strip()) > 50:  # Skip empty extracts
            documents.append(extracted_text)

    print(f"Extracted {len(documents)} clean document texts.")

    print("\n=== STEP 3: UNIFORM CHUNKING ===")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc)
        chunks.extend(doc_chunks)

    print(f"Created {len(chunks)} uniform chunks (size=500, overlap=50).")

    print(f"\n=== STEP 4: SAVING CHUNKS TO DISK ===")
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f" Saved {len(chunks)} processed chunks to: {CHUNKS_FILE}")

if __name__ == "__main__":
    main()