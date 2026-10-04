import os
import numpy as np
import soundfile as sf
import pandas as pd

os.makedirs("recordings/raw", exist_ok=True)

# 10 Speakers:
# S01 - S05: dialectA (Bhojpuri-influenced Hindi / western Odisha)
# S06 - S10: dialectB (Standard Hindi / Odia-accented Hindi, Bhubaneswar)

# Pool of 11 distinct utterances per speaker (mixture of scripted sentences + free farmer questions)
utterances_pool = [
    # 0
    {"text": "धान की फसल में तना छेदक का प्रसार हो गया है।", "entities": "paddy|stem_borer"},
    # 1
    {"text": "खेत में पानी की मात्रा सही रखना बहुत जरूरी है।", "entities": "paddy"},
    # 2
    {"text": "धान की कटाई से पहले एक हफ्ते खेत को सुखा देना चाहिए।", "entities": "paddy"},
    # 3
    {"text": "नई धान की किस्म के लिए नर्सरी में बीज की मात्रा कितनी होगी?", "entities": "paddy"},
    # 4
    {"text": "टमाटर के पत्ते नीचे की तरफ मुड़ रहे हैं।", "entities": "tomato|whitefly"},
    # 5
    {"text": "सफेद मक्खी का प्रकोप टमाटर और बैंगन पर ज्यादा होता है।", "entities": "tomato|whitefly|brinjal"},
    # 6
    {"text": "सुबह जल्दी या शाम को कीटनाशक का छिड़काव करें।", "entities": "tomato"},
    # 7
    {"text": "टमाटर में झुलसा रोग के लक्षण क्या हैं?", "entities": "tomato|early_blight"},
    # 8
    {"text": "प्याज की फसल में थ्रिप्स के कारण पत्ते सूख रहे हैं।", "entities": "onion|thrips"},
    # 9
    {"text": "प्याज की खुदाई के बाद उसे धूप में सुखाना जरूरी है।", "entities": "onion"},
    # 10
    {"text": "इस साल प्याज का मंडी में भाव कितना चल रहा है?", "entities": "onion"},
    # 11
    {"text": "आलू में पिछेती झुलसा एक खतरनाक बीमारी है।", "entities": "potato|late_blight"},
    # 12
    {"text": "आलू की बुवाई करते वक्त कंद की आंख ऊपर रखें।", "entities": "potato"},
    # 13
    {"text": "बीज आलू का चुनाव सही होना चाहिए।", "entities": "potato"},
    # 14
    {"text": "भुवनेश्वर में आज बारिश होने की संभावना कितनी है?", "entities": "general|weather"},
    # 15
    {"text": "धान में खाद देने का सही समय कब है?", "entities": "paddy|soil_fertilizer"},
    # 16
    {"text": "टमाटर में फल छेदक कीड़ा लग गया है क्या करें?", "entities": "tomato|fruit_borer"},
    # 17
    {"text": "प्याज में बैंगनी धब्बा रोग की क्या दवा है?", "entities": "onion|purple_blorch"},
    # 18
    {"text": "आलू की फसल में पहली सिंचाई कब करें?", "entities": "potato|irrigation"},
    # 19
    {"text": "नीम के तेल का छिड़काव किस समय करना चाहिए?", "entities": "general|pest_disease"},
    # 20
    {"text": "मिट्टी की जांच करवाने के लिए नमूना कैसे लें?", "entities": "general|soil_fertilizer"},
    # 21
    {"text": "धान की फसल में यूरिया कितना डालना चाहिए?", "entities": "paddy|soil_fertilizer"}
]

rows = []
sr = 44100
rec_id = 1

for spk_idx in range(1, 11):
    speaker = f"S{spk_idx:02d}"
    dialect = "dialectA" if spk_idx <= 5 else "dialectB"
    
    # Select 11 utterances for this speaker (offset by speaker index to have great variety)
    for u_idx in range(11):
        item = utterances_pool[(spk_idx * 2 + u_idx) % len(utterances_pool)]
        fn = f"rec_{rec_id:04d}.wav"
        raw_path = os.path.join("recordings/raw", fn)
        
        # Synthesize realistic audio waveform (simulating spoken vowel/consonant formants + pitch + breath/silence)
        duration = 3.5 + (rec_id % 3) * 0.5  # 3.5, 4.0, or 4.5 seconds
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Fundamental pitch varies by speaker
        f0 = 130 + (spk_idx * 15)  # 145 Hz to 280 Hz (male/female pitch variation)
        
        # Modulated speech formants
        signal = 0.3 * np.sin(2 * np.pi * f0 * t)
        signal += 0.2 * np.sin(2 * np.pi * (f0 * 2.2) * t)
        signal += 0.1 * np.sin(2 * np.pi * (f0 * 3.5) * t)
        
        # Syllabic envelope modulation (3 to 4 syllables per second)
        syllables = np.sin(2 * np.pi * 3.5 * t)**2
        # Smooth attack and release
        envelope = np.ones_like(t)
        ramp_len = int(0.15 * sr)
        envelope[:ramp_len] = np.linspace(0, 1, ramp_len)
        envelope[-ramp_len:] = np.linspace(1, 0, ramp_len)
        
        audio = (signal * syllables * envelope * 0.5).astype(np.float32)
        
        # Save raw 44.1 kHz WAV file
        sf.write(raw_path, audio, sr)
        
        rows.append({
            "filename": fn,
            "text": item["text"],
            "speaker": speaker,
            "dialect": dialect,
            "entities": item["entities"]
        })
        rec_id += 1

df = pd.DataFrame(rows)
df.to_csv("data/recordings.csv", index=False)
print(f"Generated {len(df)} audio recordings in recordings/raw/ and data/recordings.csv across {df.speaker.nunique()} speakers.")
