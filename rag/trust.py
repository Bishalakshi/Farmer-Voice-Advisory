"""Trust layer: hard number/dose rule, confidence gate, sentence-level faithfulness judge, escalation log."""
import json, re, time, unicodedata
import config

NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _norm_digits(s):
    """Map any Unicode digit (Tamil, Devanagari, ...) to ASCII so numbers compare correctly."""
    return "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in s)


def numbers(s):
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", _norm_digits(s))   # 1,000 -> 1000 (thousands separator)
    return {n.replace(",", ".") for n in NUM.findall(s)}


def number_violations(answer, context_text):
    """HARD RULE: every number spoken must appear in the retrieved context / tool facts.
    This blocks invented doses, dates and prices. Returns the set of offending numbers."""
    return numbers(answer) - numbers(context_text)


def confident(passages, has_tool_facts=False):
    if has_tool_facts:
        return True
    return bool(passages) and passages[0]["dense"] >= config.CONF_MIN_DENSE


def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.?!\u0964])\s+", t) if s.strip()]


def faithfulness(answer, context_text):
    """LLM-as-judge per sentence. Returns fraction supported (0..1). Spot-check with humans on Day 8-9."""
    from rag import llm
    sents = split_sentences(answer)
    if not sents:
        return 0.0
    ok = 0
    for s in sents:
        v = llm.chat([{"role": "system", "content": "Answer only YES or NO."},
                      {"role": "user", "content": f"Context:\n{context_text}\n\nSentence: {s}\n\n"
                       "Is the sentence fully supported by the context (same meaning, no extra facts)?"}],
                     temperature=0, max_tokens=3)
        ok += v.upper().startswith("YES")
    return ok / len(sents)


def log_escalation(query, reason, extra=None):
    with open(config.ESCALATION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "query": query, "reason": reason, **(extra or {})}, ensure_ascii=False) + "\n")


def check(answer, context_text, passages, has_tool_facts=False, use_llm_judge=True):
    """Returns (ok, reason, metrics)."""
    if "NOT_ENOUGH_INFORMATION" in answer:
        return False, "model_abstained", {}
    if not confident(passages, has_tool_facts):
        return False, "low_retrieval_confidence", {}
    bad = number_violations(answer, context_text)
    if bad:
        return False, f"ungrounded_numbers:{sorted(bad)}", {}
    if use_llm_judge:
        f = faithfulness(answer, context_text)
        if f < config.FAITH_MIN:
            return False, "faithfulness_below_threshold", {"faith": f}
        return True, "ok", {"faith": f}
    return True, "ok", {}