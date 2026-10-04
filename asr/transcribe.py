"""Whisper wrapper. variant='base' = off-the-shelf, 'ft' = our merged LoRA model."""
import os, sys
import pandas as pd
import soundfile as sf
import config


def load_asr(variant="base", device=None):
    import torch
    from transformers import pipeline
    device = device if device is not None else (0 if torch.cuda.is_available() else -1)
    name = config.WHISPER_BASE if variant == "base" else config.WHISPER_FT_DIR
    if variant != "base" and not os.path.isdir(name):
        raise FileNotFoundError(f"{name} not found - run asr/finetune_lora.py first")
    return pipeline("automatic-speech-recognition", model=name, device=device,
                    chunk_length_s=30,
                    generate_kwargs={"language": config.LANG_NAME.lower(), "task": "transcribe",
                                     "max_new_tokens": 128})


def transcribe(asr, path):
    audio, sampling_rate = sf.read(path, dtype="float32")
    return asr({"raw": audio, "sampling_rate": sampling_rate})["text"].strip()


if __name__ == "__main__":
    # python -m asr.transcribe base data/manifest.csv out_base.csv
    variant, manifest, out = sys.argv[1:4]
    df = pd.read_csv(manifest)
    df = df[df.split == "test"].copy()
    asr = load_asr(variant)
    df["hyp"] = [transcribe(asr, p) for p in df.audio_path]
    df.to_csv(out, index=False)
    print("saved", out, len(df))
