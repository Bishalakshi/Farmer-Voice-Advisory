"""Text normalisation + MMS-TTS synthesis."""
import re
import numpy as np, scipy.io.wavfile as wavfile
import config

_model = _tok = None


def normalize(text, lang=config.LANG):
    text = re.sub(r"[*_#`>\[\]]", "", text)
    text = text.replace("%", " percent ")
    try:                                        # digits -> words if num2words supports the language
        from num2words import num2words
        text = re.sub(r"\d+(?:\.\d+)?", lambda m: num2words(float(m.group()) if "." in m.group() else int(m.group()), lang=lang), text)
    except Exception:
        pass                                    # keep digits; CHECK by listening that they are spoken
    return re.sub(r"\s+", " ", text).strip()


def synth(text, out_path="out.wav", lang=config.LANG):
    global _model, _tok
    import torch
    from transformers import VitsModel, AutoTokenizer
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(config.MMS_TTS[lang])
        _model = VitsModel.from_pretrained(config.MMS_TTS[lang])
    inputs = _tok(normalize(text, lang), return_tensors="pt")
    with torch.no_grad():
        wav = _model(**inputs).waveform[0].numpy()
    wavfile.write(out_path, _model.config.sampling_rate, (wav * 32767).astype(np.int16))
    return out_path
