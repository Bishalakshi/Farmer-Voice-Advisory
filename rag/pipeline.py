"""End-to-end orchestrator. Flags let us run baselines B1-B4 and ablations from the same code."""
import json, time
import config
from messages import ABSTAIN, WHY
from rag import nlu, generator, trust
from rag.risk import risk_check


class Advisor:
    def __init__(self, asr_variant="ft", use_rag=True, use_trust=True, use_lexicon=True, use_judge=True):
        self.asr_variant, self.use_rag, self.use_trust = asr_variant, use_rag, use_trust
        self.use_lexicon, self.use_judge = use_lexicon, use_judge
        self.ontology = json.load(open(config.ONTOLOGY_FILE, encoding="utf-8"))
        self._asr = self._ret = self._vocab = None

    # lazy loaders keep start-up light for text-only evaluations
    @property
    def asr(self):
        from asr.transcribe import load_asr
        self._asr = self._asr or load_asr(self.asr_variant); return self._asr

    @property
    def retriever(self):
        from rag.retriever import Retriever
        self._ret = self._ret or Retriever(); return self._ret

    @property
    def vocab(self):
        from asr.lexicon_fix import load_vocab
        self._vocab = self._vocab or load_vocab(); return self._vocab

    def ask_audio(self, path, **kw):
        from asr.transcribe import transcribe
        from asr.lexicon_fix import fix_transcript
        t0 = time.time()
        text = transcribe(self.asr, path)
        if self.use_lexicon:
            text, _ = fix_transcript(text, self.vocab)
        out = self.ask_text(text, **kw)
        out["t_asr"] = time.time() - t0 - out["t_text"]
        out["transcript"] = text
        return out

    def ask_text(self, text, place=None, lat=None, lon=None):
        t0 = time.time()

        # Day 6 safety guard: dose, medicine, mixing and poisoning questions never reach the LLM.
        # Only active when the trust layer is on, so baseline B1 stays unchanged.
        if self.use_trust:
            hit = risk_check(text)
            if hit:
                trust.log_escalation(text, "risk:" + hit[0])
                return {"answer": hit[1], "raw_answer": "", "ok": False, "reason": "risk:" + hit[0],
                        "nlu": {"intent": "risk"}, "metrics": {}, "passages": [],
                        "t_text": time.time() - t0}

        u = nlu.extract(text, self.ontology)
        facts = []
        try:                                     # tools need internet; offline -> silently fall back to KB only
            if u["intent"] in ("weather", "crop_sowing") or "rain" in (u.get("query_en") or "").lower():
                from tools import weather
                g = weather.geocode(place or u.get("location") or "") if (place or u.get("location")) else None
                if g:
                    facts.append(weather.forecast_facts(g[0], g[1], g[2]))
                elif lat is not None:
                    facts.append(weather.forecast_facts(lat, lon, "your location"))
        except Exception as e:
            print("weather tool unavailable:", e)
        if u["intent"] == "market_price" and u.get("crop"):
            try:
                from tools import market
                f = market.price_facts(u["crop"]); facts += [f] if f else []
            except Exception as e:
                print("market tool unavailable:", e)
        passages = []
        if self.use_rag:
            passages = self.retriever.search([text, u["query_en"]], crop=u.get("crop"))
        ctx_passages = [{"text": f, "source": "tool", "dense": 1.0} for f in facts] + passages
        ctx_text = "\n".join(p["text"] for p in ctx_passages)
        ans = generator.generate(u["query_en"], text, ctx_passages, use_rag=self.use_rag)
        ok, reason, met = True, "not_checked", {}
        if self.use_trust and self.use_rag:
            ok, reason, met = trust.check(ans, ctx_text, passages, has_tool_facts=bool(facts), use_llm_judge=self.use_judge)
        if not ok:
            trust.log_escalation(text, reason, {"nlu": u})
            final = ABSTAIN[config.LANG]
        else:
            src = (passages[0]["source"] if passages else "tool")
            final = ans + (" " + WHY[config.LANG].format(src=src) if self.use_rag else "")
        return {"answer": final, "raw_answer": ans, "ok": ok, "reason": reason, "nlu": u, "metrics": met,
                "passages": [{k: p[k] for k in ("id", "dense", "source") if k in p} for p in passages],
                "t_text": time.time() - t0}