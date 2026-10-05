import pandas as pd, jiwer
from asr.lexicon_fix import load_vocab, fix_transcript
from eval.eval_asr import norm, entity_error_rate

syn = pd.read_csv("data/train_base_synth.csv").fillna("")
fle = pd.read_csv("data/train_base_fleurs.csv").fillna("")
vocab = load_vocab()

def wer(df, hyps):
    return jiwer.wer([norm(x) for x in df.text], [norm(x) for x in hyps])

base_eer = entity_error_rate(syn.entities, syn.hyp)
base_ws, base_wf = wer(syn, syn.hyp), wer(fle, fle.hyp)
print(f"no fix : EER {base_eer:.3f} | synth WER {base_ws:.3f} | FLEURS WER {base_wf:.3f}\n")
print("cutoff   EER   synthWER  FLEURS_WER")

best = None
for c in range(70, 100, 5):
    hs = [fix_transcript(h, vocab, cutoff=c)[0] for h in syn.hyp]
    hf = [fix_transcript(h, vocab, cutoff=c)[0] for h in fle.hyp]
    e, ws, wf = entity_error_rate(syn.entities, hs), wer(syn, hs), wer(fle, hf)
    safe = wf <= base_wf + 0.005 and ws <= base_ws + 0.005
    print(f"{c:>5}  {e:.3f}   {ws:.3f}    {wf:.3f}   {'ok' if safe else 'hurts WER'}")
    if safe and (best is None or e <= best[1]):
        best = (c, e)
print("\nchosen CUTOFF:", best[0] if best else "none safe, use 95 or no fix")