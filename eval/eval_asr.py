"""WER / CER / Entity Error Rate, overall and per dialect.
Usage: python -m eval.eval_asr out_base.csv out_ft.csv [--fix]"""
import sys, pandas as pd, jiwer
from asr.lexicon_fix import load_vocab, fix_transcript


def entity_error_rate(refs_entities, hyps):
    tot = miss = 0
    for ents, hyp in zip(refs_entities, hyps):
        for e in str(ents).split("|"):
            if e and e != "nan":
                tot += 1
                miss += e.lower() not in str(hyp).lower()
    return miss / tot if tot else float("nan")


def report(df, name):
    rows = []
    for d, g in [("ALL", df)] + list(df.groupby("dialect")):
        rows.append({"system": name, "dialect": d, "n": len(g),
                     "WER": jiwer.wer(g.text.tolist(), g.hyp.tolist()),
                     "CER": jiwer.cer(g.text.tolist(), g.hyp.tolist()),
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
            df2 = df.copy(); df2["hyp"] = [fix_transcript(h, vocab)[0] for h in df.hyp]
            out.append(report(df2, f + "+lexicon"))
    res = pd.concat(out)
    print(res.round(3).to_string(index=False)); res.to_csv("data/asr_results.csv", index=False)
