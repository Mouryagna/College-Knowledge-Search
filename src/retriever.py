import json
import re
import pickle
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import (
    CHUNKS_PATH, FAISS_PATH, BM25_PATH,
    BI_ENCODER_NAME, CROSS_ENCODER_NAME
)


class HybridRetriever:
    def __init__(self):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.faiss_index = faiss.read_index(str(FAISS_PATH))

        with open(BM25_PATH, "rb") as f:
            self.bm25_index = pickle.load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bi_encoder = SentenceTransformer(BI_ENCODER_NAME, device=self.device)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME, max_length=512, device=self.device)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def search_bm25(self, query: str, top_k: int = 5):
        scores = self.bm25_index.get_scores(self._tokenize(query))
        top_ids = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[int(i)] for i in top_ids]

    def search_faiss(self, query: str, top_k: int = 5):
        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype("float32")
        _, top_ids = self.faiss_index.search(q_emb, top_k)
        return [self.chunks[int(i)] for i in top_ids[0]]

    def retrieve(self, query: str, candidate_k: int = 20, top_k: int = 5, rrf_k: int = 60, ce_threshold: float = -1.0):
        # 1. Dense candidates
        q_emb = self.bi_encoder.encode([query], normalize_embeddings=True).astype("float32")
        _, dense_ids = self.faiss_index.search(q_emb, candidate_k)

        # 2. Sparse candidates
        bm25_scores = self.bm25_index.get_scores(query.lower().split())
        sparse_ids = np.argsort(bm25_scores)[::-1][:candidate_k]

        # 3. Reciprocal Rank Fusion
        rrf = {}
        for rank, idx in enumerate(dense_ids[0]):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse_ids):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (rrf_k + rank + 1)

        fused_ids = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:candidate_k]

        # 4. Cross-Encoder Re-Ranking
        pairs = [(query, self.chunks[i]["chunk_text"]) for i in fused_ids]
        ce_scores = self.cross_encoder.predict(pairs)

        ranked = sorted(zip(fused_ids, ce_scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank_idx, (doc_id, score) in enumerate(ranked):
            if rank_idx == 0 or score >= ce_threshold:
                item = self.chunks[doc_id].copy()
                item["rrf_score"] = round(float(rrf[doc_id]), 4)
                item["ce_score"] = round(float(score), 4)
                results.append(item)
        return results