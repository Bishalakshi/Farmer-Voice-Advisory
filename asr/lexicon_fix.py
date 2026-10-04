"""Post-ASR correction: snap near-miss words to known crop/pest/chemical names from the ontology."""
import json, re
from rapidfuzz import process, fuzz
import config


def load_vocab(path=config.ONTOLOGY_FILE):
    onto = json.load(open(path, encoding="utf-8"))
    vocab = {}                                   # surface form -> canonical name
    for group in ("crops", "pests"):
        for canon, info in onto.get(group, {}).items():
            for s in info["synonyms"] + [canon]:
                vocab[s.lower()] = canon
    return vocab


def fix_transcript(text, vocab, cutoff=85):
    """Replace 1-word and 2-word spans that fuzzily match a vocabulary term (score >= cutoff).
    Exact matches are left untouched. Returns (new_text, list_of_replacements)."""
    if not vocab:
        return text, []
    keys = list(vocab)
    words = re.findall(r"\S+", text)
    out, i, changes = [], 0, []
    while i < len(words):
        done = False
        for n in (2, 1):                         # try bigrams first
            if i + n > len(words):
                continue
            span = " ".join(words[i:i + n])
            if span.lower() in vocab:            # already correct
                out.append(span); i += n; done = True; break
            hit = process.extractOne(span.lower(), keys, scorer=fuzz.ratio, score_cutoff=cutoff)
            if hit:
                out.append(hit[0]); changes.append((span, hit[0], hit[1])); i += n; done = True; break
        if not done:
            out.append(words[i]); i += 1
    return " ".join(out), changes
