"""Central configuration. Change LANG once; everything else follows."""
import os

LANG = os.getenv("VA_LANG", "ta")            # ta | hi | te | kn  (set to your pilot language)
LANG_NAME = {"ta": "Tamil", "hi": "Hindi", "te": "Telugu", "kn": "Kannada"}[LANG]

# ---- models (candidates: verify versions/licences before final submission) ----
WHISPER_BASE = "openai/whisper-small"
WHISPER_FT_DIR = "models/whisper-small-lora-merged"      # created on Day 4
EMBED_MODEL = "intfloat/multilingual-e5-small"           # upgrade option: BAAI/bge-m3
LLM_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"          # 4-bit GGUF, wildcard file below
LLM_FILE = "*Q4_K_M.gguf"
LLM_CTX = 4096
LLM_GPU_LAYERS = int(os.getenv("VA_GPU_LAYERS", "-1"))   # -1 = all on GPU, 0 = CPU only
NLLB_MODEL = "facebook/nllb-200-distilled-600M"          # optional translate-mode ablation (non-commercial licence)
NLLB_CODE = {"ta": "tam_Taml", "hi": "hin_Deva", "te": "tel_Telu", "kn": "kan_Knda", "en": "eng_Latn"}
MMS_TTS = {"ta": "facebook/mms-tts-tam", "hi": "facebook/mms-tts-hin",
           "te": "facebook/mms-tts-tel", "kn": "facebook/mms-tts-kan"}

# ---- paths ----
KB_FILE = "data/kb.jsonl"
KB_INDEX_DIR = "kb/index"
ONTOLOGY_FILE = "data/ontology.json"
MARKET_FILE = "data/market_prices.csv"
ESCALATION_LOG = "data/escalations.jsonl"

# ---- pipeline thresholds (tune on Day 7 using eval/tune_threshold.py) ----
TOP_K = 5
CONF_MIN_DENSE = 0.80        # min cosine of best passage; e5 scores cluster high, so MUST be tuned
FAITH_MIN = 0.99             # fraction of answer sentences the judge must accept
INTENTS = ["pest_disease", "weather", "soil_fertilizer", "market_price", "crop_sowing", "irrigation", "other"]
