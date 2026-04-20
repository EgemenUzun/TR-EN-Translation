"""
E2E Speech Translation - Improved Training Script (v2)

Key improvements over full_e2e_train.py:
  - eos_token_id correctly set (prevents infinite = generation)
  - predict_with_generate=True with BLEU compute_metrics (real quality signal)
  - Cosine LR schedule + higher warmup (bridge needs time to stabilize)
  - Entire wav2vec2 encoder frozen initially (only cross-attn bridge + decoder trains)
  - Gradient checkpointing (saves ~30% VRAM)
  - 5 epochs with early stopping via best-model selection
  - Proper num_beams=4 for eval generation
"""

import gc
import torch
import librosa
import numpy as np
import evaluate as evaluate_lib
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import (
    SpeechEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

ENCODER_ID = "facebook/wav2vec2-large-xlsr-53"
DECODER_ID = "Helsinki-NLP/opus-mt-tr-en"
OUTPUT_DIR = "./e2e_v2_results"
DATASET_PATH = "ready_covost_dataset"


@dataclass
class SpeechSeq2SeqCollator:
    extractor: Any
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.extractor.pad(input_features, padding=True, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def build_model(tokenizer: AutoTokenizer) -> SpeechEncoderDecoderModel:
    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(ENCODER_ID, DECODER_ID)

    # Copy shared vocab weights — paths differ between standard and tc-big MarianMT
    temp = AutoModelForSeq2SeqLM.from_pretrained(DECODER_ID)
    shared_weights = temp.model.shared.weight.data.clone()
    del temp
    gc.collect()

    dec = model.decoder
    if hasattr(dec, "model") and hasattr(dec.model, "decoder"):
        dec.model.decoder.embed_tokens.weight.data = shared_weights
    if hasattr(dec, "lm_head"):
        dec.lm_head.weight.data = shared_weights
    dec.tie_weights()  # re-ties all weight-shared matrices after injection

    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Scale down the enc_to_dec_proj bridge (randomly initialized).
    # Default Xavier init + large wav2vec2 activations → loss ~224k → grad NaN.
    # Small gain (0.01) brings initial loss down to the expected ~10-12 range.
    if hasattr(model, "enc_to_dec_proj"):
        torch.nn.init.xavier_normal_(model.enc_to_dec_proj.weight, gain=0.01)
        if model.enc_to_dec_proj.bias is not None:
            torch.nn.init.zeros_(model.enc_to_dec_proj.bias)

    # Freeze entire encoder — only bridge + decoder trains
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Gradient checkpointing saves ~30% VRAM
    model.decoder.gradient_checkpointing_enable()

    return model


def make_compute_metrics(tokenizer: AutoTokenizer):
    bleu_metric = evaluate_lib.load("sacrebleu")
    chrf_metric = evaluate_lib.load("chrf")

    def compute_metrics(eval_preds):
        pred_ids, label_ids = eval_preds

        # pred_ids may contain -100 (batch padding) or IDs beyond SPM's piece count
        pred_ids = np.where(pred_ids >= 0, pred_ids, tokenizer.pad_token_id)
        if hasattr(tokenizer, "sp_model"):
            spm_max = tokenizer.sp_model.get_piece_size()
            pred_ids = np.where(pred_ids < spm_max, pred_ids, tokenizer.pad_token_id)

        label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)

        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        references_wrapped = [[r] for r in references]

        bleu = bleu_metric.compute(predictions=predictions, references=references_wrapped)
        chrf = chrf_metric.compute(predictions=predictions, references=references_wrapped)
        return {
            "bleu": round(bleu["score"], 2),
            "chrf": round(chrf["score"], 2),
        }

    return compute_metrics


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("1. Loading dataset...")
    raw_dataset = load_from_disk(DATASET_PATH)

    extractor = AutoFeatureExtractor.from_pretrained(ENCODER_ID)
    tokenizer = AutoTokenizer.from_pretrained(DECODER_ID)

    def preprocess(batch):
        audio, _ = librosa.load(batch["audio_path"], sr=16000)
        batch["input_values"] = extractor(audio, sampling_rate=16000).input_values[0]
        batch["labels"] = tokenizer(batch["translation"]).input_ids
        return batch

    print("2. Preprocessing dataset (multi-process)...")
    processed = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names, num_proc=4)

    split = processed.train_test_split(test_size=0.1, seed=42)
    train_data, eval_data = split["train"], split["test"]
    print(f"   Train: {len(train_data):,} | Eval: {len(eval_data):,}")

    print("\n3. Building model...")
    model = build_model(tokenizer)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    collator = SpeechSeq2SeqCollator(extractor=extractor, tokenizer=tokenizer)

    print("\n4. Configuring training...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,          # Effective batch = 16
        learning_rate=1e-4,                     # Bridge starts from scratch, needs higher LR
        lr_scheduler_type="cosine",
        warmup_steps=1000,                      # More warmup — bridge is cold, needs gentle start
        max_grad_norm=1.0,
        num_train_epochs=5,
        bf16=True,                              # bf16 has no scaler — avoids "unscale FP16 gradients" crash with frozen encoder
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,             # Use real generation for eval BLEU
        generation_max_length=128,
        generation_num_beams=4,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",           # Select best by BLEU, not loss
        greater_is_better=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=extractor,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    import sys
    resume = next((a for a in sys.argv[1:] if a.startswith("--resume=")), None)
    resume_path = resume.split("=", 1)[1] if resume else None

    print("\n" + "=" * 55)
    print("STARTING E2E TRAINING v2 (encoder frozen, 5 epochs)")
    if resume_path:
        print(f"Resuming from: {resume_path}")
    print("=" * 55)
    trainer.train(resume_from_checkpoint=resume_path)

    print("\nSaving best model...")
    trainer.save_model(f"{OUTPUT_DIR}/best_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
    print(f"Done! Model saved to {OUTPUT_DIR}/best_model")


if __name__ == "__main__":
    main()
