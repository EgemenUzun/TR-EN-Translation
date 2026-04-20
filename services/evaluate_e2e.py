import torch
import librosa
from datasets import load_from_disk
import evaluate
from tqdm import tqdm

from e2e_common import (
    DATASET_PATH,
    E2E_BEST_MODEL_DIR,
    generation_kwargs,
    load_processors_for_inference,
    load_trained_e2e_model,
)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"1. Loading fully-trained E2E model on {device}...")

    feature_extractor, tokenizer = load_processors_for_inference(E2E_BEST_MODEL_DIR)
    model = load_trained_e2e_model(E2E_BEST_MODEL_DIR, device=device)

    print("2. Loading dataset...")
    dataset = load_from_disk(str(DATASET_PATH))
    subset_dataset = dataset.shuffle(seed=42).select(range(min(100, len(dataset))))

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    true_en_texts = []
    pred_en_texts = []

    gen_args = generation_kwargs(model)

    print("\n3. Starting evaluation...")
    for item in tqdm(subset_dataset, desc="Evaluating E2E model"):
        audio_array, _ = librosa.load(item["audio_path"], sr=16000)
        inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(inputs["input_values"], **gen_args)

        pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        pred_en_texts.append(pred_text)
        true_en_texts.append([item["translation"]])

    print("\n4. Computing final scores...")
    bleu_score = bleu_metric.compute(predictions=pred_en_texts, references=true_en_texts)
    chrf_score = chrf_metric.compute(predictions=pred_en_texts, references=true_en_texts)

    print("=" * 50)
    print("FULL END-TO-END SYSTEM EVALUATION RESULTS (100 samples)")
    print("=" * 50)
    print("ASR WER (Word Error Rate) : N/A (E2E skips Turkish text)")
    print(f"MT  BLEU score            : {bleu_score['score']:.2f} (higher is better)")
    print(f"MT  chrF score            : {chrf_score['score']:.2f} (higher is better)")
    print("=" * 50)


if __name__ == "__main__":
    main()
