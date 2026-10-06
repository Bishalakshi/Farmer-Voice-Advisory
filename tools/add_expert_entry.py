import json, sys, datetime
if len(sys.argv) != 6:
    sys.exit('usage: python -m tools.add_expert_entry <expert_name> <crop> <topic> "<question>" "<approved_answer>"')
name, crop, topic, q, ans = sys.argv[1:]
n = sum(1 for l in open("data/kb.jsonl", encoding="utf-8") if l.strip())
row = dict(id=f"E{n+1:03d}", text=f"{q} {ans}", crop=crop, topic=topic, region="general", season="all",
           stage="all", source=f"expert-approved: {name}", date=datetime.date.today().isoformat())
with open("data/kb.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("added", row["id"], "- now run: python -m kb.build_index")