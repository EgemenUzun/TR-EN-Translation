import torch
from datasets import load_from_disk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
from tqdm import tqdm

def main():
    # 1. Load the preprocessed dataset
    print("Loading dataset...")
    dataset = load_from_disk("hazir_covost_verisi")
    
    # Shuffle and select a smaller subset (100 samples) for quick evaluation
    # Set seed for reproducibility
    subset_dataset = dataset.shuffle(seed=42).select(range(100))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 2. Load Models
    print(f"Loading ASR and MT models on {device}...")
    asr_pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small", 
        device=device
    )
    
    mt_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tr-en")
    mt_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tr-en").to(device)

    # 3. Load Evaluation Metrics
    wer_metric = evaluate.load("wer")
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    # Storage for predictions and references
    asr_predictions = []
    asr_references = []
    
    mt_predictions = []
    mt_references = []

    print("\nStarting evaluation on 100 samples...")
    # 4. Evaluation Loop using tqdm for progress tracking
    for item in tqdm(subset_dataset, desc="Evaluating Cascade System"):
        audio_array = item["audio_path"]["array"]
        true_tr_text = item["sentence"]
        true_en_text = item["translation"]
        
        # --- ASR Stage ---
        asr_result = asr_pipe(audio_array, generate_kwargs={"language": "turkish", "suppress_tokens": ""})
        pred_tr_text = asr_result["text"].strip()
        
        # --- MT Stage ---
        pred_en_text = ""
        if pred_tr_text: # Ensure ASR output is not empty
            inputs = mt_tokenizer(pred_tr_text, return_tensors="pt", padding=True).to(device)
            outputs = mt_model.generate(**inputs)
            pred_en_text = mt_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Append to lists
        asr_predictions.append(pred_tr_text if pred_tr_text else " ")
        asr_references.append(true_tr_text if true_tr_text else " ")
        
        mt_predictions.append(pred_en_text if pred_en_text else " ")
        # SacreBLEU expects references as a list of lists: [[ref1, ref2, ...]]
        mt_references.append([true_en_text])

    # 5. Compute Metrics
    print("\nComputing final scores...")
    
    wer_score = wer_metric.compute(predictions=asr_predictions, references=asr_references)
    bleu_score = bleu_metric.compute(predictions=mt_predictions, references=mt_references)
    chrf_score = chrf_metric.compute(predictions=mt_predictions, references=mt_references)

    # 6. Display Results
    print("="*40)
    print("CASCADE SYSTEM EVALUATION RESULTS (100 Samples)")
    print("="*40)
    print(f"ASR WER (Word Error Rate) : {wer_score * 100:.2f}% (Lower is better)")
    print(f"MT  BLEU Score            : {bleu_score['score']:.2f} (Higher is better)")
    print(f"MT  chrF Score            : {chrf_score['score']:.2f} (Higher is better)")
    print("="*40)

if __name__ == "__main__":
    main()