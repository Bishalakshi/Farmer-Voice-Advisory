"""Query understanding: intent + entities + English rewrite. LLM JSON with ontology-keyword fallback."""
import json, re
import config
from rag import llm

SYSTEM = f"""You extract structured data from a farmer's spoken question (transcribed, may contain errors, may mix {config.LANG_NAME} and English).
Return ONLY a JSON object with keys:
"intent": one of {config.INTENTS},
"crop": crop name in English or null,
"stage": growth stage or null,
"symptom": short English description of the problem or null,
"location": place name mentioned or null,
"query_en": the question rewritten in clear English."""


def keyword_fallback(text, ontology):
    low = text.lower()
    crop = next((c for c, i in ontology["crops"].items() if any(s in low for s in i["synonyms"])), None)
    intent = "other"
    if any(w in low for w in ["rain", "weather", "forecast"]): intent = "weather"
    elif any(w in low for w in ["price", "rate", "market", "sell"]): intent = "market_price"
    elif any(w in low for w in ["pest", "insect", "leaf", "disease", "curl", "spot"]): intent = "pest_disease"
    return {"intent": intent, "crop": crop, "stage": None, "symptom": None, "location": None, "query_en": text}


def extract(text, ontology):
    try:
        raw = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                       temperature=0, max_tokens=200, json_mode=True)
        d = json.loads(raw)
        if d.get("intent") not in config.INTENTS:
            d["intent"] = "other"
        d.setdefault("query_en", text)
        return d
    except Exception:
        return keyword_fallback(text, ontology)
