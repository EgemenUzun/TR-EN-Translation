import torch
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_from_disk
from transformers import (
    SpeechEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from e2e_common import (
    DATASET_PATH,
    DECODER_MODEL_ID,
    E2E_BEST_MODEL_DIR,
    E2E_OUTPUT_DIR,
    ENCODER_MODEL_ID,
    configure_e2e_generation_config,
    load_feature_extractor,
    load_tokenizer,
    seq2seq_trainer_processing_kwarg,
    seq2seq_training_eval_strategy_kwarg,
    training_precision_kwargs,
)


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
    dataset = load_from_disk(str(DATASET_PATH))

    feature_extractor = load_feature_extractor()
    tokenizer = load_tokenizer()

    def prepare_dataset(batch):
        audio_array, _ = librosa.load(batch["audio_path"], sr=16000)
        batch["input_values"] = feature_extractor(audio_array, sampling_rate=16000).input_values[0]
        batch["labels"] = tokenizer(
            batch["translation"],
            truncation=True,
            max_length=256,
        ).input_ids
        return batch

    print("2. Mapping dataset features (This might take a few minutes for 11k+ samples)...")
    processed_dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=4)

    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset["train"]
    eval_data = split_dataset["test"]

    print(f"Total Training Samples: {len(train_data)}")
    print(f"Total Validation Samples: {len(eval_data)}")

    print("\n3. Constructing the E2E SpeechEncoderDecoderModel...")
    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
        ENCODER_MODEL_ID,
        DECODER_MODEL_ID,
    )
    configure_e2e_generation_config(model, tokenizer)
    model.freeze_feature_encoder()
    model.to(device)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=feature_extractor,
        tokenizer=tokenizer,
    )

    print("\n4. Configuring full training...")
    precision = training_precision_kwargs()
    eval_kw = seq2seq_training_eval_strategy_kwarg("epoch")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(E2E_OUTPUT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=500,
        num_train_epochs=10,
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        **eval_kw,
        **precision,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        **seq2seq_trainer_processing_kwarg(tokenizer),
    )

    print("\n" + "=" * 50)
    print("STARTING FULL END-TO-END TRAINING")
    print("=" * 50)

    trainer.train()

    E2E_BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("\nTraining finished. Saving best checkpoint...")
    trainer.save_model(str(E2E_BEST_MODEL_DIR))
    tokenizer.save_pretrained(str(E2E_BEST_MODEL_DIR))
    feature_extractor.save_pretrained(str(E2E_BEST_MODEL_DIR))
    print(f"Model, tokenizer, and feature extractor saved to {E2E_BEST_MODEL_DIR}")


if __name__ == "__main__":
    main()