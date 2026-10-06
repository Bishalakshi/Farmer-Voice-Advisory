"""Digits -> Hindi words for TTS (num2words has no Hindi). Have a Hindi speaker check the table."""
import re, unicodedata

_W = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
      "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस "
      "चालीस इकतालीस बयालीस तैंतालीस चौवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ "
      "साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उनासी "
      "अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()
assert len(_W) == 100


def int_to_hi(n):
    if n < 100: return _W[n]
    if n < 1000:
        h, r = divmod(n, 100)
        return _W[h] + " सौ" + (" " + _W[r] if r else "")
    if n < 100000:
        t, r = divmod(n, 1000)
        return _W[t] + " हज़ार" + (" " + int_to_hi(r) if r else "")
    return " ".join(_W[int(d)] for d in str(n))


def _ascii(s):
    return "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in s)


def numbers_to_hindi(text):
    text = _ascii(text)
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    text = text.replace("%", " प्रतिशत ")
    def conv(m):
        ip, _, fp = m.group().partition(".")
        out = int_to_hi(int(ip))
        if fp: out += " दशमलव " + " ".join(_W[int(d)] for d in fp)
        return out
    return re.sub(r"\d+(?:\.\d+)?", conv, text)