"""Run systems B1-B4 on the same question set; writes a CSV for expert rating + prints auto metrics.
Input CSV columns: question,gold_ids,crop,expected_behaviour(answer|abstain),place
Usage: python -m eval.eval_e2e data/e2e_test.csv"""
import sys, time, pandas as pd, psutil, os
from rag.pipeline import Advisor

SYSTEMS = {"B1_noRAG": dict(use_rag=False, use_trust=False),
           "B3_noTrust": dict(use_rag=True, use_trust=False),
           "B4_full": dict(use_rag=True, use_trust=True),
           "B4_noJudge": dict(use_rag=True, use_trust=True, use_judge=False)}
df = pd.read_csv(sys.argv[1]).fillna("")
rows = []
for name, kw in SYSTEMS.items():
    adv = Advisor(**kw)
    for i, r in df.iterrows():
        t = time.time(); out = adv.ask_text(r.question, place=r.place or None); dt = time.time() - t
        rows.append({"system": name, "qid": i, "question": r.question, "expected": r.expected_behaviour,
                     "answer": out["answer"], "abstained": not out["ok"], "reason": out["reason"],
                     "faith": out["metrics"].get("faith"), "latency_s": dt,
                     "expert_correct_1to5": "", "expert_safe_yes_no": ""})   # <- agronomists fill these two columns
res = pd.DataFrame(rows); res.to_csv("data/e2e_for_rating.csv", index=False)
g = res.assign(should_abstain=res.expected.eq("abstain"))
summ = g.groupby("system").apply(lambda d: pd.Series({
    "n": len(d), "abstain_rate": d.abstained.mean(),
    "wrong_abstain_on_answerable": d[~d.should_abstain].abstained.mean(),
    "answered_when_should_abstain": (~d[d.should_abstain].abstained).mean() if d.should_abstain.any() else float("nan"),
    "latency_p50": d.latency_s.median(), "latency_p95": d.latency_s.quantile(.95)}))
print(summ.round(3).to_string()); print("RAM MB:", psutil.Process(os.getpid()).memory_info().rss // 2**20)
