import torch
import librosa
from datasets import load_from_disk
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
import evaluate
from tqdm import tqdm

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"1. Loading Fully-Trained E2E Model on {device}...")

    model_path = "./e2e_full_results/best_model"
    encoder_id = "facebook/wav2vec2-large-xlsr-53"
    decoder_id = "Helsinki-NLP/opus-mt-tr-en"

    # Load processors
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)

    # Eğitilmiş modeli yükle
    model = SpeechEncoderDecoderModel.from_pretrained(model_path).to(device)

    # --- HAYAT KURTARAN DÜZELTME (THE LIFESAVER FIX) ---
    print("Applying LM Head weights fix...")
    model.decoder.lm_head.weight = model.decoder.model.decoder.embed_tokens.weight
    # ---------------------------------------------------

    print("2. Loading dataset...")
    dataset = load_from_disk("ready_covost_dataset")
    # Kaskad ile adil karşılaştırma için aynı 100 veriyi çek
    subset_dataset = dataset.shuffle(seed=42).select(range(100))

    # Metrikler
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    true_en_texts = []
    pred_en_texts = []

    print("\n3. Starting Evaluation on 100 samples...")
    for item in tqdm(subset_dataset, desc="Evaluating E2E Model"):
        audio_array, _ = librosa.load(item["audio_path"], sr=16000)
        inputs = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_values"],
                max_length=200,
                num_beams=4,
                decoder_start_token_id=tokenizer.pad_token_id
            )
        
        pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        pred_en_texts.append(pred_text)
        true_en_texts.append([item["translation"]])

    print("\n4. Computing final scores...")
    bleu_score = bleu_metric.compute(predictions=pred_en_texts, references=true_en_texts)
    chrf_score = chrf_metric.compute(predictions=pred_en_texts, references=true_en_texts)

    # Sonuç Ekranı
    print("="*50)
    print("FULL END-TO-END SYSTEM EVALUATION RESULTS (100 Samples)")
    print("="*50)
    print("ASR WER (Word Error Rate) : N/A (E2E skips Turkish text!)")
    print(f"MT  BLEU Score            : {bleu_score['score']:.2f} (Higher is better)")
    print(f"MT  chrF Score            : {chrf_score['score']:.2f} (Higher is better)")
    print("="*50)

if __name__ == "__main__":
    main()