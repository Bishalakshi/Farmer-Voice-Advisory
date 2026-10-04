"""System Usability Scale scoring. CSV: one row per participant, columns q1..q10 with answers 1-5.
Usage: python -m eval.sus data/sus_responses.csv   (target: mean >= 68)"""
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
odd = [f"q{i}" for i in (1, 3, 5, 7, 9)]; even = [f"q{i}" for i in (2, 4, 6, 8, 10)]
score = ((df[odd] - 1).sum(axis=1) + (5 - df[even]).sum(axis=1)) * 2.5
print(f"n={len(df)}  SUS mean={score.mean():.1f}  sd={score.std():.1f}  min={score.min():.1f}  max={score.max():.1f}")
