"""Recall@k, MRR for dense / bm25 / hybrid. CSV columns: question,gold_ids (| separated),crop
Usage: python -m eval.eval_retrieval data/retrieval_test.csv"""
import sys, pandas as pd, numpy as np
from rag.retriever import Retriever

df = pd.read_csv(sys.argv[1]).fillna("")
R = Retriever()
rows = []
for mode in ("dense", "bm25", "hybrid"):
    for use_filter in (False, True):
        rec5, rr = [], []
        for _, r in df.iterrows():
            gold = set(r.gold_ids.split("|"))
            hits = R.search([r.question], k=5, crop=(r.crop or None) if use_filter else None, mode=mode)
            ids = [h["id"] for h in hits]
            rec5.append(len(gold & set(ids)) / len(gold))
            rr.append(next((1 / (i + 1) for i, x in enumerate(ids) if x in gold), 0))
        rows.append({"mode": mode, "crop_filter": use_filter, "Recall@5": np.mean(rec5), "MRR": np.mean(rr)})
out = pd.DataFrame(rows); print(out.round(3).to_string(index=False)); out.to_csv("data/retrieval_results.csv", index=False)
