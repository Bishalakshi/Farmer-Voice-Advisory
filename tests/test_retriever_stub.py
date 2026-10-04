import json, numpy as np
from rank_bm25 import BM25Okapi
from rag.retriever import Retriever, tok


class FakeEnc:                       # bag-of-words hashing encoder stands in for multilingual-e5
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.zeros(64)
            for w in tok(t.split(": ", 1)[-1]): v[hash(w) % 64] += 1
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.array(out)


def make():
    chunks = [json.loads(l) for l in open("data/kb_sample.jsonl")]
    r = Retriever.__new__(Retriever)
    r.chunks = chunks; r.model = FakeEnc()
    r.emb = r.model.encode(["passage: " + c["text"] for c in chunks])
    r.bm25 = BM25Okapi([tok(c["text"]) for c in chunks])
    return r


def test_search_returns_relevant_and_respects_crop_filter():
    r = make()
    hits = r.search(["whitefly on tomato leaves yellow sticky traps"], k=3, crop="tomato")
    assert hits[0]["id"] in ("S001", "S007")
    assert all(h["crop"] in ("tomato", "general") for h in hits)
    assert {"dense", "score"} <= set(hits[0])


def test_modes_run():
    r = make()
    for m in ("dense", "bm25", "hybrid"):
        assert len(r.search(["paddy stem borer"], k=2, mode=m)) == 2
