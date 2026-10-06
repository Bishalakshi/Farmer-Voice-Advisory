"""Demo UI. Run: python app.py"""
import json, time
import gradio as gr
import config
from rag.pipeline import Advisor
from tts.speak import synth

adv = Advisor(asr_variant="ft")


def log_interaction(consent, transcript, answer, ok, reason):
    if not consent:
        return
    with open("data/interaction_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(ts=time.time(), q=transcript, a=answer, ok=ok, reason=reason), ensure_ascii=False) + "\n")


def handle(audio_path, place, consent):
    r = adv.ask_audio(audio_path, place=place or None)
    log_interaction(consent, r["transcript"], r["answer"], r["ok"], r["reason"])
    wav = synth(r["answer"], "reply.wav")
    return r["transcript"], r["answer"], wav, {k: r[k] for k in ("ok", "reason", "nlu", "metrics", "passages")}


with gr.Blocks(title="Farmer's Voice-to-Advisory") as demo:
    gr.Markdown(f"## Farmer's Voice-to-Advisory ({config.LANG_NAME})")
    mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak your question")
    place = gr.Textbox(label="Village / district (optional, used for weather)")
    consent = gr.Checkbox(label="I agree my question and the answer text may be logged for evaluation (audio is never stored)", value=False)
    btn = gr.Button("Get advice")
    t, a, w, dbg = gr.Textbox(label="What I heard"), gr.Textbox(label="Advice"), gr.Audio(label="Spoken advice"), gr.JSON(label="Debug")
    btn.click(handle, [mic, place, consent], [t, a, w, dbg])

if __name__ == "__main__":
    demo.launch()