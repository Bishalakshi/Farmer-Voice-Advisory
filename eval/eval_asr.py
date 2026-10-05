"""WER / CER / Entity Error Rate, overall and per group (the 'dialect' column).
Usage: python -m eval.eval_asr out_base.csv out_ft.csv [--fix]"""
import sys, re, json, pandas as pd, jiwer
from asr.lexicon_fix import load_vocab, fix_transcript

CUTOFF = 85          # Day 5: replace with the value tuned on TRAIN data only

ONTO = json.load(open("data/ontology.json", encoding="utf-8"))
SYN = {}
for g in ("crops", "pests"):
    for canon, info in ONTO[g].items():
        SYN[canon.replace(" ", "_")] = [s.lower() for s in info["synonyms"] + [canon]]


def norm(s):
    """Remove punctuation so Whisper's '।' '?' do not count as word errors."""
    s = re.sub(r"[।\.,\?!\"'()\-–:;]", " ", str(s))
    return re.sub(r"\s+", " ", s).strip().lower()


def entity_error_rate(refs_entities, hyps):
    tot = miss = 0
    for ents, hyp in zip(refs_entities, hyps):
        for e in str(ents).split("|"):
            if not e or e == "nan":
                continue
            tot += 1
            forms = SYN.get(e, [e.replace("_", " ")])
            miss += not any(f in str(hyp).lower() for f in forms)
    return miss / tot if tot else float("nan")


def report(df, name):
    rows = []
    for d, g in [("ALL", df)] + list(df.groupby("dialect")):
        ref = [norm(x) for x in g.text]; hyp = [norm(x) for x in g.hyp]
        rows.append({"system": name, "dialect": d, "n": len(g),
                     "WER": jiwer.wer(ref, hyp), "CER": jiwer.cer(ref, hyp),
                     "EER": entity_error_rate(g.entities, g.hyp)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    files = [f for f in sys.argv[1:] if f.endswith(".csv")]
    fix = "--fix" in sys.argv
    vocab = load_vocab()
    out = []
    for f in files:
        df = pd.read_csv(f).fillna("")
        out.append(report(df, f))
        if fix:
            df2 = df.copy()
            df2["hyp"] = [fix_transcript(h, vocab, cutoff=CUTOFF)[0] for h in df.hyp]
            out.append(report(df2, f + "+lexicon"))
    res = pd.concat(out)
    print(res.round(3).to_string(index=False)); res.to_csv("data/asr_results.csv", index=False)
