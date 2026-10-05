import os, numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly
from tts.speak import synth

rng = np.random.default_rng(7)
sheet = pd.read_csv("data/recording_sheet.csv").fillna("")
sent = sheet.drop_duplicates("text")[["text", "entities", "split"]].reset_index(drop=True)

def noise(snr):
       def f(y):
           p = np.mean(y ** 2) + 1e-9
           return y + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(y))
       return f
def phone(y):
       return resample_poly(resample_poly(y, 1, 2), 2, 1)
VARIANTS = {"clean": lambda y: y, "noise10": noise(10), "noise5": noise(5), "phone8k": phone,
               "slow": lambda y: resample_poly(y, 10, 9), "fast": lambda y: resample_poly(y, 9, 10)}
os.makedirs("recordings/raw_synth", exist_ok=True); os.makedirs("recordings/wav16k", exist_ok=True)
rows, n = [], 0
for i, s in sent.iterrows():
       base = f"recordings/raw_synth/base_{i:03d}.wav"
       synth(s.text, base)
       y, sr = sf.read(base, dtype="float32")
       if sr != 16000:
           y = resample_poly(y, 16000, sr)
       for vname, fn in VARIANTS.items():
           z = fn(y).astype("float32"); z = z / max(1e-6, np.abs(z).max()) * 0.9
           n += 1; fn_out = f"rec_{n:04d}.wav"; path = f"recordings/wav16k/{fn_out}"
           sf.write(path, z, 16000)
           rows.append(dict(filename=fn_out, audio_path=path, text=s.text, speaker="mms_hi",
                            dialect="synth_" + vname, entities=s.entities, split=s.split, condition=vname))
df = pd.DataFrame(rows)
df.to_csv("data/manifest.csv", index=False, encoding="utf-8")
print(df.groupby(["split", "dialect"]).size().unstack(fill_value=0))
print("text overlap train/test:", len(set(df[df.split == "train"].text) & set(df[df.split == "test"].text)))
