import torch
import librosa
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk
from transformers import (
    SpeechEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# --- 1. DATA COLLATOR FOR SPEECH-TO-SEQUENCE ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.tokenizer.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels

        return batch

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("1. Loading FULL dataset...")
    dataset = load_from_disk("ready_covost_dataset")
    
    encoder_id = "facebook/wav2vec2-large-xlsr-53"
    decoder_id = "Helsinki-NLP/opus-mt-tr-en"

    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)

    # --- 2. DATA PREPROCESSING FUNCTION ---
    def prepare_dataset(batch):
        audio_array, _ = librosa.load(batch["audio_path"], sr=16000)
        batch["input_values"] = feature_extractor(audio_array, sampling_rate=16000).input_values[0]
        batch["labels"] = tokenizer(batch["translation"]).input_ids
        return batch

    print("2. Mapping dataset features (This might take a few minutes for 11k+ samples)...")
    processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=4)
    
    # Standard 90/10 split for deep learning
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]
    
    print(f"Total Training Samples: {len(train_data)}")
    print(f"Total Validation Samples: {len(eval_data)}")

    # --- 3. MODEL ARCHITECTURE SETUP ---
    print("\n3. Constructing the E2E SpeechEncoderDecoderModel...")
    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_id, decoder_id)

    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Freeze the CNN layers to save memory and focus on the cross-attention bridge
    model.freeze_feature_encoder()
    model.to(device)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=feature_extractor, tokenizer=tokenizer)

    # --- 4. FULL TRAINING ARGUMENTS ---
    print("\n4. Configuring Full Training Hyperparameters...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./e2e_full_results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective Batch Size = 16
        learning_rate=1e-4,             # Slightly higher LR for faster bridge convergence
        warmup_steps=500,               # Give the model time to adapt before large updates
        num_train_epochs=10,             # Cross-attention bridge needs several epochs to converge
        bf16=True,                      # RTX 40-series magic enabled
        eval_strategy="epoch",          # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save checkpoint at the end of each epoch
        logging_steps=50,               # Show progress every 50 steps
        predict_with_generate=False,    # Disabled to drastically speed up training
        save_total_limit=2,             # Keep only the last 2 models to save disk space
        load_best_model_at_end=True,    # Automatically load the model with the lowest loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none"
    )

    # --- 5. INITIALIZE TRAINER ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=feature_extractor,
        data_collator=data_collator,
    )

    print("\n" + "="*50)
    print("STARTING FULL END-TO-END TRAINING (Grab a coffee, this will take a while!)")
    print("="*50)
    
    trainer.train()

    print("\nTraining Finished! Saving the absolute best model...")
    trainer.save_model("./e2e_full_results/best_model")
    print("You have successfully trained a complete End-to-End Speech Translation model!")

if __name__ == "__main__":
    main()