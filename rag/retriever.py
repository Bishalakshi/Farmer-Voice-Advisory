"""Hybrid retriever: dense (multilingual-e5) + BM25, fused with Reciprocal Rank Fusion, with metadata filter."""
import json, re
import numpy as np
import config


def tok(t):
    return re.findall(r"\w+", t.lower())


def rrf_fuse(dense, bm25, k=60):
    """Pure-numpy Reciprocal Rank Fusion over two score vectors."""
    rd = np.empty(len(dense)); rd[np.argsort(-dense)] = np.arange(len(dense))
    rb = np.empty(len(bm25));  rb[np.argsort(-bm25)] = np.arange(len(bm25))
    return 1.0 / (k + rd + 1) + 1.0 / (k + rb + 1)


class Retriever:
    def __init__(self, index_dir=config.KB_INDEX_DIR):
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi
        self.chunks = json.load(open(f"{index_dir}/chunks.json", encoding="utf-8"))
        self.emb = np.load(f"{index_dir}/emb.npy")
        self.model = SentenceTransformer(config.EMBED_MODEL)
        self.bm25 = BM25Okapi([tok(c["text"]) for c in self.chunks])

    def search(self, queries, k=config.TOP_K, crop=None, mode="hybrid"):
        """queries: list of strings (original-language query + English rewrite). Best score over variants."""
        qv = self.model.encode(["query: " + q for q in queries], normalize_embeddings=True)
        dense = (qv @ self.emb.T).max(axis=0)
        bm = np.max([self.bm25.get_scores(tok(q)) for q in queries], axis=0)
        score = {"hybrid": rrf_fuse(dense, bm), "dense": dense, "bm25": bm}[mode]
        allowed = np.array([(crop is None) or c["crop"] in (crop, "general") for c in self.chunks])
        score = np.where(allowed, score, -1e9)
        idx = np.argsort(-score)[:k]
        return [{**self.chunks[i], "dense": float(dense[i]), "score": float(score[i])} for i in idx if allowed[i]]
