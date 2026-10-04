import warnings

import numpy as np, pandas as pd
from rag.trust import number_violations, confident, split_sentences, numbers
from rag.retriever import rrf_fuse
from kb.build_index import chunk_text, ingest_checks
from asr.lexicon_fix import fix_transcript
from asr.prepare_manifest import speaker_split
from eval.eval_asr import entity_error_rate
from tools.crop_validation import run, FEATS


def test_numbers_blocked_when_not_in_context():
    ctx = "Rain chance 80 percent. Rainfall 12.5 mm."
    assert number_violations("There is 80 percent chance of rain.", ctx) == set()
    assert number_violations("Spray 5 ml per litre.", ctx) == {"5"}


def test_devanagari_and_tamil_digits_normalised():
    assert numbers("\u0968\u0966 \u0BE8\u0BE6") == {"20"}      # Devanagari 20, Tamil 20 -> same "20"


def test_confidence_gate():
    assert confident([{"dense": 0.9}])
    assert not confident([{"dense": 0.5}])
    assert not confident([])
    assert confident([], has_tool_facts=True)


def test_split_sentences():
    assert len(split_sentences("One. Two? Three!")) == 3


def test_rrf_prefers_item_good_in_both():
    d = np.array([0.9, 0.8, 0.1]); b = np.array([0.1, 5.0, 4.0])
    assert rrf_fuse(d, b).argmax() == 1


def test_chunking_and_ingest():
    long = " ".join(f"Sentence number {i} about crops." for i in range(60))
    ch = chunk_text(long, max_words=40, overlap=5)
    assert len(ch) > 3 and all(len(c.split()) < 60 for c in ch)
    rows = [{"id": "a", "text": "same text", "crop": "x", "topic": "t", "region": "r", "season": "s", "stage": "st", "source": "s", "date": "d"},
            {"id": "b", "text": "Same  text!", "crop": "x", "topic": "t", "region": "r", "season": "s", "stage": "st", "source": "s", "date": "d"},
            {"id": "c", "text": "other", "crop": "", "topic": "t", "region": "r", "season": "s", "stage": "st", "source": "s", "date": "d"}]
    p = ingest_checks(rows)
    assert any("duplicate" in m for _, m in p) and any("missing" in m for _, m in p)


def test_lexicon_fix_snaps_near_miss_only():
    vocab = {"whitefly": "whitefly", "white fly": "whitefly", "tomato": "tomato"}
    t, ch = fix_transcript("my tomatto has whitefy problem", vocab)
    assert "tomato" in t and "whitefly" in t and len(ch) == 2
    t2, ch2 = fix_transcript("my tomato is fine", vocab)
    assert ch2 == []


def test_speaker_disjoint_split():
    df = pd.DataFrame({"speaker": [f"s{i%10}" for i in range(50)]})
    out = speaker_split(df, 0.3)
    tr = set(out[out.split == "train"].speaker); te = set(out[out.split == "test"].speaker)
    assert tr.isdisjoint(te) and len(te) == 3


def test_entity_error_rate():
    assert entity_error_rate(["tomato|whitefly"], ["tomato leaf"]) == 0.5


def test_crop_validation_runs_on_synthetic_data():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(8):
        c = rng.normal(size=7) * 10 + k * 15
        for _ in range(60):
            rows.append(list(c + rng.normal(size=7) * 3) + [f"crop{k}"])
    df = pd.DataFrame(rows, columns=FEATS + ["label"])
    df[["temperature", "humidity", "rainfall"]] = df[["temperature", "humidity", "rainfall"]].abs()
    out = run(df, verbose=False)
    assert out.loc["RandomForest", "random_test_acc"] > 0.9
    cov, size = out.loc["RandomForest", "conformal_random_cov_setsize"]
    assert cov >= 0.8 and size >= 1.0


def test_crop_validation_no_runtime_warning_on_all_unseen_groups():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(8):
        c = rng.normal(size=7) * 10 + k * 15
        for _ in range(60):
            rows.append(list(c + rng.normal(size=7) * 3) + [f"crop{k}"])
    df = pd.DataFrame(rows, columns=FEATS + ["label"])
    df[["temperature", "humidity", "rainfall"]] = df[["temperature", "humidity", "rainfall"]].abs()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        run(df, verbose=False)
