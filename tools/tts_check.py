import os, pandas as pd
from tts.speak import synth

ANSWERS = [
 "कल बारिश की संभावना 80% है।",
 "टमाटर में सफेद मक्खी के लिए पीले चिपचिपे जाल लगाएँ।",
 "धान की बुवाई जून से जुलाई में करें।",
 "आलू की पत्तियों पर धब्बे दिखें तो कृषि विज्ञान केंद्र से संपर्क करें।",
 "प्याज का आज का भाव 25 रुपये प्रति किलो है।",
 "तापमान 32 डिग्री सेल्सियस रहेगा।",
 "टमाटर में हर 7 दिन में निगरानी करें।",
 "क्षमा करें, मेरे पास इसका पक्का उत्तर नहीं है।",
 "बारिश 12.5 मिलीमीटर हो सकती है।",
 "छिड़काव के बाद हाथ साबुन से धोएँ।",
]
os.makedirs("recordings/tts_check", exist_ok=True)
rows = []
for i, t in enumerate(ANSWERS, 1):
    p = f"recordings/tts_check/tts_{i:02d}.wav"
    synth(t, p)
    rows.append(dict(filename=os.path.basename(p), audio_path=p, text=t, speaker="mms_hi",
                     dialect="tts_check", entities="", split="test", condition="tts"))
pd.DataFrame(rows).to_csv("data/manifest_tts_check.csv", index=False, encoding="utf-8")
print("wrote", len(rows))