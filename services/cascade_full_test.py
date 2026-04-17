import torch
from datasets import load_from_disk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    # 1. Load the prepared dataset
    print("Loading preprocessed dataset...")
    dataset = load_from_disk("ready_covost_dataset")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load ASR Model (Speech-to-Text)
    print("Loading ASR model (Whisper-Small)...")
    asr_pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-small", 
        device=device
    )

    # 3. Load MT Model (Text-to-Text Translation)
    # We use explicit Model and Tokenizer for better control over generation parameters
    print("Loading MT model (Helsinki-NLP Turkish-to-English)...")
    mt_model_name = "Helsinki-NLP/opus-mt-tr-en"
    mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_name)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_name).to(device)

    print("\n" + "="*50)
    print("STARTING FULL CASCADE SYSTEM TEST (End-to-End)")
    print("="*50 + "\n")

    # 4. Run test on the first 5 samples
    for i in range(5):
        audio_sample = dataset[i]["audio_path"]["array"]
        true_tr = dataset[i]["sentence"]
        true_en = dataset[i]["translation"]
        
        # --- Stage 1: Automatic Speech Recognition (ASR) ---
        asr_result = asr_pipe(
            audio_sample, 
            generate_kwargs={"language": "turkish", "suppress_tokens": ""}
        )
        predicted_tr = asr_result["text"].strip()
        
        # --- Stage 2: Machine Translation (MT) ---
        predicted_en = ""
        if predicted_tr:
            # Tokenize the ASR output
            inputs = mt_tokenizer(predicted_tr, return_tensors="pt", padding=True).to(device)
            # Generate translation
            outputs = mt_model.generate(**inputs)
            # Decode the generated tokens back to text
            predicted_en = mt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # --- Results Output ---
        print(f"Sample #{i+1}")
        print(f"Actual Turkish   : {true_tr}")
        print(f"ASR Transcription: {predicted_tr}")
        print(f"Cascade Translation (EN): {predicted_en}")
        print(f"Ground Truth Target (EN): {true_en}")
        print("-" * 50)

if __name__ == "__main__":
    main()