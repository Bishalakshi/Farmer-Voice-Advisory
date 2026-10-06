import re, sys, pandas as pd
from tts.hindi_numbers import numbers_to_hindi, _ascii

sys.stdout.reconfigure(encoding="utf-8")

d = pd.read_csv("out_tts_check.csv").fillna("")
ok, lines = 0, []
for i, r in d.iterrows():
    nums = re.findall(r"\d+(?:\.\d+)?", _ascii(r.text))
    hyp = _ascii(r.hyp)
    miss = [n for n in nums if n not in hyp and numbers_to_hindi(n).split()[0] not in hyp]
    status = "OK" if not miss else f"MISSING {miss}"
    ok += not miss
    lines.append(f"{i+1:>2} {status:<18} IN: {r.text}\n   OUT: {r.hyp}")
summary = f"\n{ok}/{len(d)} clips kept their numbers"
print("\n".join(lines) + summary)
open("data/tts_check_results.txt", "w", encoding="utf-8").write("\n".join(lines) + summary + "\n")
print("saved data/tts_check_results.txt")