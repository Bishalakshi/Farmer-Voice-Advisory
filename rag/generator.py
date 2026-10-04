"""Grounded answer generation."""
import config
from rag import llm

SYS_GROUNDED = """You are an agricultural advisory assistant for smallholder farmers.
Rules:
1. Use ONLY the numbered context passages below. If they do not answer the question, reply exactly: NOT_ENOUGH_INFORMATION
2. Never write any number, dose, quantity, chemical name or date that is not in the context.
3. Answer in {lang}, in at most 4 short, simple sentences suitable for speaking aloud. No lists, no markdown, no citations.
4. If a pesticide or dose is needed but not in the context, tell the farmer to ask the local agriculture officer."""

SYS_FREE = "You are an agricultural advisory assistant. Answer in {lang} in at most 4 short spoken sentences."


def format_context(passages):
    return "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))


def generate(question_en, question_orig, passages, use_rag=True):
    if use_rag:
        sys = SYS_GROUNDED.format(lang=config.LANG_NAME)
        user = f"Context:\n{format_context(passages)}\n\nQuestion: {question_orig}\n(English: {question_en})"
    else:                                     # baseline B1: no retrieval
        sys = SYS_FREE.format(lang=config.LANG_NAME)
        user = question_orig
    return llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}], temperature=0.1)
