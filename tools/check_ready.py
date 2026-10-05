import os, pandas as pd
m = pd.read_csv("data/manifest.csv")
tr, te = m[m.split == "train"], m[m.split == "test"]
print("train/test rows:", len(tr), len(te), "(expect 96 / 36)")
print("text overlap:", len(set(tr.text) & set(te.text)), "(expect 0)")
print("backslash paths:", m.audio_path.str.contains("\\", regex=False).sum(), "(expect 0)")
print("missing files:", sum(not os.path.exists(p) for p in m.audio_path), "(expect 0)")
for f in ["data/manifest_fleurs_test.csv", "data/fleurs_train.csv", "out_base.csv", "out_base_fleurs.csv"]:
    print(f, "exists" if os.path.exists(f) else "MISSING")
if os.path.exists("out_base.csv"):
    b = pd.read_csv("out_base.csv"); print("baseline rows:", len(b), "(expect 36)"); print(b.hyp.head(3).tolist())