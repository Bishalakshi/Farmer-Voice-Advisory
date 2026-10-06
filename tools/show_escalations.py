import json, os
p = "data/escalations.jsonl"
L = [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []
print(len(L), "escalations")
for i, x in enumerate(L):
    print(i, x["reason"], "|", x["query"])