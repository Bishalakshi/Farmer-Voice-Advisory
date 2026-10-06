"""Text normalisation + MMS-TTS synthesis."""
import re
import numpy as np, scipy.io.wavfile as wavfile
import config

_model = _tok = None

# "%" must become a word in the TARGET language (MMS-TTS ignores English letters). Native speaker: confirm.
PERCENT = {"ta": "சதவீதம்", "hi": "प्रतिशत", "te": "శాతం", "kn": "ಶೇಕಡಾ"}

NUM = re.compile(r"\d+(?:\.\d+)?")


def _num2words_fallback(text, lang):
    """Digits -> words via num2words if it supports the language; otherwise leave digits unchanged."""
    try:
        from num2words import num2words
    except ImportError:
        return text

    def conv(m):
        s = m.group()
        try:
            return num2words(float(s) if "." in s else int(s), lang=lang)
        except NotImplementedError:      # language not supported by num2words
            return s
    return NUM.sub(conv, text)


def normalize(text, lang=config.LANG):
    text = re.sub(r"[*_#`>\[\]]", "", text)
    text = text.replace("%", " " + PERCENT.get(lang, "percent") + " ")

    if lang == "hi":
        try:
            from tts.hindi_numbers import numbers_to_hindi
            text = numbers_to_hindi(text)
        except ImportError:                       # helper module missing -> generic fallback
            text = _num2words_fallback(text, lang)
    else:
        text = _num2words_fallback(text, lang)    # digits stay if unsupported; CHECK by listening

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