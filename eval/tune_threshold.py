"""Pick CONF_MIN_DENSE: for answerable questions (gold_ids non-empty) vs out-of-scope questions (gold_ids empty),
sweep thresholds on top-1 dense cosine. Pick the threshold with the best F1 for 'answerable'.
Usage: python -m eval.tune_threshold data/threshold_val.csv   (same columns as retrieval test; gold_ids empty = should abstain)"""
import sys, pandas as pd, numpy as np
from rag.retriever import Retriever

df = pd.read_csv(sys.argv[1]).fillna("")
R = Retriever()
top = np.array([R.search([q], k=1, crop=None)[0]["dense"] for q in df.question])
y = np.array([bool(g) for g in df.gold_ids])
best = (0, None)
for t in np.arange(0.6, 0.95, 0.01):
    pred = top >= t
    tp = (pred & y).sum(); fp = (pred & ~y).sum(); fn = (~pred & y).sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    if f1 > best[0]: best = (f1, t)
print(f"best F1={best[0]:.3f} at CONF_MIN_DENSE={best[1]:.2f}  -> put this in config.py")
