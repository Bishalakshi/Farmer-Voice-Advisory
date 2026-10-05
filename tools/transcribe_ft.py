import sys, torch, pandas as pd, soundfile as sf
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

MODEL = "openai/whisper-small"        # must match finetune.py
adapter, manifest, out = sys.argv[1:4]
proc = WhisperProcessor.from_pretrained(MODEL, language="hindi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda")
if adapter != "none":
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.eval()

df = pd.read_csv(manifest).fillna("")
hyps = []
for i in range(0, len(df), 8):
    audio = [sf.read(p, dtype="float32")[0] for p in df.audio_path[i:i + 8]]
    feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
    with torch.no_grad():
        ids = model.generate(input_features=feats, language="hindi", task="transcribe", max_new_tokens=225)
    hyps += proc.batch_decode(ids, skip_special_tokens=True)
df["hyp"] = [h.strip() for h in hyps]
df.to_csv(out, index=False, encoding="utf-8-sig")
print("saved", out, len(df))
