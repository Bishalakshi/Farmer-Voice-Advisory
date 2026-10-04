"""Ingest checks + embedding index. Usage: python -m kb.build_index [data/kb.jsonl]"""
import json, os, re, sys
import numpy as np
import config

REQUIRED = ["id", "text", "crop", "topic", "region", "season", "stage", "source", "date"]


def chunk_text(text, max_words=110, overlap=15):
    """Sentence-aware chunking for long documents (split on . ? ! and the Devanagari/Indic danda)."""
    sents = re.split(r"(?<=[.?!\u0964])\s+", text.strip())
    chunks, cur = [], []
    for s in sents:
        if sum(len(x.split()) for x in cur) + len(s.split()) > max_words and cur:
            chunks.append(" ".join(cur))
            tail = " ".join(" ".join(cur).split()[-overlap:])
            cur = [tail] if overlap else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def ingest_checks(rows):
    problems, seen = [], {}
    for r in rows:
        miss = [k for k in REQUIRED if not str(r.get(k, "")).strip()]
        if miss:
            problems.append((r.get("id"), f"missing {miss}"))
        key = re.sub(r"\W+", " ", r.get("text", "").lower()).strip()
        if key in seen:
            problems.append((r.get("id"), f"duplicate of {seen[key]}"))
        seen[key] = r.get("id")
    return problems


def main(path=config.KB_FILE):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    probs = ingest_checks(rows)
    for p in probs:
        print("INGEST WARNING:", p)
    if probs:
        print("Fix warnings (or consciously accept) before trusting results.")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL)
    emb = model.encode(["passage: " + r["text"] for r in rows], normalize_embeddings=True, show_progress_bar=True)
    os.makedirs(config.KB_INDEX_DIR, exist_ok=True)
    np.save(f"{config.KB_INDEX_DIR}/emb.npy", emb)
    json.dump(rows, open(f"{config.KB_INDEX_DIR}/chunks.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("indexed", len(rows), "chunks ->", config.KB_INDEX_DIR)


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
