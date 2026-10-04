"""Build train/test manifest with SPEAKER-DISJOINT split.
Input : data/recordings_template.csv style file (filename,text,speaker,dialect,entities)
Output: data/manifest.csv with an added 'split' column.
Usage : python -m asr.prepare_manifest data/recordings.csv recordings/ --test_frac 0.3
"""
import argparse, os, random
import pandas as pd
import librosa, soundfile as sf


def speaker_split(df, test_frac=0.3, seed=42):
    """Whole speakers go to test, so the model never hears a test speaker in training."""
    rng = random.Random(seed)
    spk = sorted(df.speaker.unique())
    rng.shuffle(spk)
    n_test = max(1, round(len(spk) * test_frac))
    test_spk = set(spk[:n_test])
    df = df.copy()
    df["split"] = df.speaker.map(lambda s: "test" if s in test_spk else "train")
    return df


def to_16k_mono(src, dst):
    y, _ = librosa.load(src, sr=16000, mono=True)
    sf.write(dst, y, 16000)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("audio_dir")
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--test_frac", type=float, default=0.3)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    os.makedirs("recordings/wav16k", exist_ok=True)
    paths = []
    for fn in df.filename:
        dst = os.path.join("recordings/wav16k", os.path.splitext(fn)[0] + ".wav")
        to_16k_mono(os.path.join(a.audio_dir, fn), dst)
        paths.append(dst)
    df["audio_path"] = paths
    df = speaker_split(df, a.test_frac)
    df.to_csv(a.out, index=False)
    print(df.groupby(["split", "dialect"]).size())
