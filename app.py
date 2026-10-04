"""Demo UI. Run: python app.py   (Gradio gives a mic button; share=True gives a public link for the demo)."""
import gradio as gr
import config
from rag.pipeline import Advisor
from tts.speak import synth

adv = Advisor(asr_variant="ft")


def handle(audio_path, place):
    r = adv.ask_audio(audio_path, place=place or None)
    wav = synth(r["answer"], "reply.wav")
    return r["transcript"], r["answer"], wav, {k: r[k] for k in ("ok", "reason", "nlu", "metrics", "passages")}


with gr.Blocks(title="Farmer's Voice-to-Advisory") as demo:
    gr.Markdown(f"## Farmer's Voice-to-Advisory ({config.LANG_NAME})")
    mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak your question")
    place = gr.Textbox(label="Village / district (optional, used for weather)")
    btn = gr.Button("Get advice")
    t, a, w, dbg = gr.Textbox(label="What I heard"), gr.Textbox(label="Advice"), gr.Audio(label="Spoken advice"), gr.JSON(label="Debug")
    btn.click(handle, [mic, place], [t, a, w, dbg])

if __name__ == "__main__":
    demo.launch()
