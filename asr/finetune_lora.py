"""LoRA fine-tune Whisper-small on our manifest (train split only). Run on a Colab T4 / any 16GB GPU.
Usage: python -m asr.finetune_lora --manifest data/manifest.csv --extra_manifest data/public_subset.csv --epochs 5
'--extra_manifest' is optional: a CSV with the same columns built from a public corpus subset
(e.g. Common Voice / IndicVoices / Kathbath) to add speaker variety."""
import argparse
from dataclasses import dataclass
from typing import Any
import pandas as pd, torch
from datasets import Dataset, Audio
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model
import config


@dataclass
class Collator:
    processor: Any
    start_id: int

    def __call__(self, feats):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in feats], return_tensors="pt")
        lab = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in feats], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.start_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--extra_manifest", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--out", default=config.WHISPER_FT_DIR)
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    df = df[df.split == "train"][["audio_path", "text"]]
    if a.extra_manifest:
        df = pd.concat([df, pd.read_csv(a.extra_manifest)[["audio_path", "text"]]])
    print("training utterances:", len(df))

    proc = WhisperProcessor.from_pretrained(config.WHISPER_BASE, language=config.LANG_NAME, task="transcribe")
    ds = Dataset.from_dict({"audio": df.audio_path.tolist(), "text": df.text.tolist()}).cast_column("audio", Audio(16000))

    def prep(ex):
        au = ex["audio"]
        ex["input_features"] = proc.feature_extractor(au["array"], sampling_rate=au["sampling_rate"]).input_features[0]
        ex["labels"] = proc.tokenizer(ex["text"]).input_ids
        return ex
    ds = ds.map(prep, remove_columns=ds.column_names)

    model = WhisperForConditionalGeneration.from_pretrained(config.WHISPER_BASE)
    model.config.forced_decoder_ids = None
    model.generation_config.language = config.LANG_NAME.lower()
    model.generation_config.task = "transcribe"
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"], bias="none"))
    model.print_trainable_parameters()

    args = Seq2SeqTrainingArguments(
        output_dir="models/whisper_lora_ckpt", per_device_train_batch_size=a.bs, gradient_accumulation_steps=2,
        learning_rate=a.lr, warmup_steps=20, num_train_epochs=a.epochs, fp16=torch.cuda.is_available(),
        logging_steps=10, save_strategy="no", remove_unused_columns=False, label_names=["labels"],
        report_to="none")
    Seq2SeqTrainer(model=model, args=args, train_dataset=ds,
                   data_collator=Collator(proc, model.config.decoder_start_token_id)).train()

    merged = model.merge_and_unload()               # single standalone model for fast inference
    merged.save_pretrained(a.out); proc.save_pretrained(a.out)
    print("saved merged model to", a.out)


if __name__ == "__main__":
    main()
