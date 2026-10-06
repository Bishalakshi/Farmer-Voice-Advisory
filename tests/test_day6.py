import pytest
from rag.trust import numbers, number_violations, check
from rag.risk import risk_check
from tts.hindi_numbers import numbers_to_hindi


def test_thousands_separator():
    assert numbers("1,000 और 2.5") == {"1000", "2.5"}
    assert number_violations("1,000 मिमी", "1000 मिमी") == set()


def test_check_blocks_each_failure():
    ok, r, _ = check("5 मिली मिलाएँ", "नीम तेल", [{"dense": 0.9}], use_llm_judge=False)
    assert not ok and r.startswith("ungrounded_numbers")
    assert check("NOT_ENOUGH_INFORMATION", "x", [{"dense": 0.9}])[1] == "model_abstained"
    assert check("ठीक है", "x", [{"dense": 0.1}], use_llm_judge=False)[1] == "low_retrieval_confidence"
    assert check("ठीक है", "x", [{"dense": 0.9}], use_llm_judge=False)[0]


def test_sentence_judge(monkeypatch):
    monkeypatch.setattr("rag.llm.chat", lambda *a, **k: "NO")
    ok, r, _ = check("ठीक है।", "x", [{"dense": 0.9}])
    assert not ok and r == "faithfulness_below_threshold"
    monkeypatch.setattr("rag.llm.chat", lambda *a, **k: "YES")
    assert check("ठीक है।", "x", [{"dense": 0.9}])[0]


@pytest.mark.parametrize("q,exp", [
    ("टमाटर में इमिडाक्लोप्रिड कितनी मात्रा में डालूँ?", "dose"),
    ("मेरी गाय को बुखार है, कौन सी दवा दूँ?", "human_or_animal_medicine"),
    ("कीटनाशक निगल लिया गया है", "poisoning"),
    ("दो अलग-अलग कीटनाशक आपस में मिला सकते हैं क्या?", "mixing"),
])
def test_risk_blocks(q, exp):
    assert risk_check(q)[0] == exp


def test_safe_questions_not_blocked():
    assert risk_check("टमाटर में सफेद मक्खी का इलाज क्या है?") is None
    assert risk_check("कल बारिश होगी या नहीं?") is None


def test_hindi_numbers():
    assert "अस्सी प्रतिशत" in numbers_to_hindi("80%")
    assert numbers_to_hindi("32") == "बत्तीस"
    assert "दशमलव" in numbers_to_hindi("12.5")