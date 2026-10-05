import random, pandas as pd

IGNORE = {"general", "irrigation", "pest_disease", "soil_fertilizer", "weather"}
ALIAS = {"purple_blorch": "purple_blotch"}
CROPS = ["paddy", "tomato", "onion", "potato"]
N_HOLDOUT, N_NOISY_TRAIN, N_NOISY_TEST, SEED = 6, 4, 3, 7

old = pd.read_csv("data/manifest.csv").fillna("")

def clean(e):
    out = []
    for x in str(e).split("|"):
        x = ALIAS.get(x, x)
        if x and x not in IGNORE and x not in out:
            out.append(x)
    return "|".join(out)

old["entities"] = old.entities.map(clean)
sent = old.drop_duplicates("text")[["text", "entities"]].reset_index(drop=True)
rng = random.Random(SEED)

# pick hold-out (test-only) sentences: one per crop first, then fill up to 6
hold = []
for c in CROPS:
    pool = [i for i in sent.index if c in sent.entities[i].split("|") and i not in hold]
    if pool:
        hold.append(rng.choice(pool))
rest = [i for i in sent.index if i not in hold]
while len(hold) < N_HOLDOUT:
    hold.append(rest.pop(rng.randrange(len(rest))))
train_idx = [i for i in sent.index if i not in hold]

# one row per recording
spk = old.drop_duplicates("speaker")[["speaker", "dialect", "split"]].sort_values("speaker")
rows = []
for _, s in spk.iterrows():
    idx = hold if s.split == "test" else train_idx
    nn = N_NOISY_TEST if s.split == "test" else N_NOISY_TRAIN
    for cond, items in (("quiet", idx), ("noisy", idx[:nn])):
        for i in items:
            rows.append(dict(text=sent.text[i], speaker=s.speaker, dialect=s.dialect,
                             entities=sent.entities[i], split=s.split, condition=cond))

df = pd.DataFrame(rows)
df.insert(0, "filename", [f"rec_{n:04d}.wav" for n in range(1, len(df) + 1)])
df.to_csv("data/recording_sheet.csv", index=False, encoding="utf-8-sig")

print(df.groupby(["split", "condition"]).size())
print("text overlap train/test:", len(set(df[df.split == "train"].text) & set(df[df.split == "test"].text)))