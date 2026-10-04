# Farmer's Voice-to-Advisory - 10-day implementation kit

Voice (dialect) -> ASR -> NLU -> tools + hybrid RAG -> 4-bit LLM -> trust layer -> TTS.

## Setup (Day 1)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python      # GPU; drop CMAKE_ARGS for CPU
export VA_LANG=ta        # ta | hi | te | kn
pytest -q                # 12 logic tests must pass (no model downloads needed)
```
Free GPU: Google Colab T4 or Kaggle. Needs a Hugging Face internet connection for first model download.

## Order of running
1. `python -m asr.prepare_manifest data/recordings.csv recordings/raw`     (Day 2)
2. `cp data/kb_sample.jsonl data/kb.jsonl` then extend; `python -m kb.build_index`   (Day 2-3)
3. `python -m asr.transcribe base data/manifest.csv out_base.csv` ; `python -m eval.eval_asr out_base.csv`   (Day 3)
4. `python -m tools.crop_validation data/Crop_recommendation.csv`   (Day 3)
5. `python -m asr.finetune_lora --manifest data/manifest.csv`  (Day 4)  ->  `python -m asr.transcribe ft data/manifest.csv out_ft.csv`
6. `python -m eval.eval_asr out_base.csv out_ft.csv --fix`   (Day 5)
7. `python -m eval.eval_retrieval data/retrieval_test.csv`   (Day 3/8)
8. `python -m eval.tune_threshold data/threshold_val.csv`   (Day 7) -> edit CONF_MIN_DENSE in config.py
9. `python app.py`   (Day 7 demo)
10. `python -m eval.eval_e2e data/e2e_test.csv` ; `python -m eval.sus data/sus_responses.csv`   (Day 8-9)

## Notes
- data/kb_sample.jsonl and data/ontology.json are SAMPLE placeholders. Replace with verified content and local-language synonyms.
- messages.py strings must be verified by a native speaker.
- Model names are candidates; check current versions and licences (NLLB and some datasets are non-commercial).
