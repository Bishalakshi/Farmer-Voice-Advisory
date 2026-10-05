import torch, pandas as pd, soundfile as sf
from dataclasses import dataclass
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from peft import LoraConfig, get_peft_model

MODEL, OUT, EPOCHS = "openai/whisper-small", "adapters/ft_v1", 4

proc = WhisperProcessor.from_pretrained(MODEL, language="hindi", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL)
model.config.forced_decoder_ids = None
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"], bias="none"))
model.print_trainable_parameters()

fl = pd.read_csv("data/fleurs_train.csv").fillna("")
sy = pd.read_csv("data/manifest.csv").fillna("")
sy = sy[sy.split == "train"]
train = pd.concat([fl, sy])[["audio_path", "text"]].reset_index(drop=True)
print("train rows:", len(train), "(expect 496)")

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(train)
    def __getitem__(self, i):
        y, sr = sf.read(train.audio_path[i], dtype="float32")
        f = proc.feature_extractor(y, sampling_rate=16000).input_features[0]
        return {"input_features": f, "labels": proc.tokenizer(train.text[i]).input_ids}

@dataclass
class Collate:
    def __call__(self, b):
        x = proc.feature_extractor.pad([{"input_features": e["input_features"]} for e in b], return_tensors="pt")
        l = proc.tokenizer.pad([{"input_ids": e["labels"]} for e in b], return_tensors="pt")
        lab = l["input_ids"].masked_fill(l.attention_mask.ne(1), -100)
        if (lab[:, 0] == model.config.decoder_start_token_id).all():
            lab = lab[:, 1:]
        x["labels"] = lab
        return x

args = Seq2SeqTrainingArguments(
    output_dir="tmp_ft", per_device_train_batch_size=8, gradient_accumulation_steps=2,
    learning_rate=5e-4, warmup_steps=10, num_train_epochs=EPOCHS, fp16=True,
    logging_steps=10, save_strategy="no", report_to="none",
    remove_unused_columns=False, label_names=["labels"])
Seq2SeqTrainer(model=model, args=args, train_dataset=DS(), data_collator=Collate()).train()
model.save_pretrained(OUT)
print("saved", OUT)
