"""Pre-LLM safety guard: dose, medicine, mixing and poisoning questions never reach the generator."""
import re
EMERGENCY = "यह आपातकालीन स्थिति हो सकती है। तुरंत नज़दीकी अस्पताल या डॉक्टर के पास जाएँ।"
REFER = "इस बारे में मैं सलाह नहीं दे सकता। कृपया अपने कृषि अधिकारी या कृषि विज्ञान केंद्र से पूछें।"
RULES = [
    ("unsafe_chemical", REFER, r"खाली\s*डिब्बा|प्रतिबंधित|ज़हरीला\s*रसायन|जहरीला\s*रसायन"),
    ("poisoning", EMERGENCY, r"निगल|ज़हर|जहर|चक्कर|बेहोश|कीटनाशक\s*लग\s*गया"),
    ("human_or_animal_medicine", EMERGENCY, r"सिरदर्द|गोली|इंजेक्शन|बुखार|गाय को|बकरी को|पत्नी को|बच्चे को"),
    ("mixing", REFER, r"मिला(?:कर|ने|ऊँ|ऊं)|मिला\s*सकते|मिश्रण|एक\s*टंकी"),
    ("dose", REFER, r"कितन[ाी]\s*(?:मात्रा|ग्राम|मिली|किलो)|प्रति\s*(?:लीटर|एकड़|हेक्टेयर)|ग्राम\s*प्रति|कितन[ाी]\s+\S+\s+(?:डालूँ|डालूं|डालें|डालना)"),
]


def risk_check(text):
    for name, msg, pat in RULES:
        if re.search(pat, text):
            return name, msg
    return None