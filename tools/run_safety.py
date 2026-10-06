import json, pandas as pd
from rag.pipeline import Advisor
from rag.trust import numbers

USE_JUDGE = False          # True adds one LLM call per sentence; slow on CPU
chunks = {c["id"]: c["text"] for c in json.load(open("kb/index/chunks.json", encoding="utf-8"))}
a = Advisor(use_judge=USE_JUDGE)
df = pd.read_csv("data/safety_set.csv")
rows = []
for q in df.question:
    r = a.ask_text(q)
    ctx = " ".join(chunks.get(p["id"], "") for p in r["passages"])
    bad = sorted(numbers(r["raw_answer"]) - numbers(ctx)) if r["ok"] else []
    tool_intent = r["nlu"].get("intent") in ("weather", "crop_sowing", "market_price")
    rows.append(dict(ok=r["ok"], reason=r["reason"], spoken=r["answer"],
                     ungrounded_numbers="|".join(bad), check_by_hand=bool(bad and tool_intent),
                     human_safe_yes_no=""))
out = pd.concat([df, pd.DataFrame(rows)], axis=1)
out.to_csv("data/safety_set_results.csv", index=False, encoding="utf-8-sig")
print("questions:", len(out))
print(out.reason.str.split(":").str[0].value_counts().to_string())
print("spoken answers with ungrounded numbers:", (out.ungrounded_numbers != "").sum(), "(target 0)")