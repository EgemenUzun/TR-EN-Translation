import torch
import librosa
from datasets import load_from_disk
from transformers import pipeline

def main():
    # 1. Load the prepared dataset
    print("Loading preprocessed dataset...")
    dataset = load_from_disk("ready_covost_dataset")

    # 2. Setup device (GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 3. Initialize Whisper model (using 'small' for better Turkish accuracy)
    print("Loading Whisper 'large-v3' model...")
    asr_pipe = pipeline(
        "automatic-speech-recognition", 
        model="openai/whisper-large-v3", 
        device=device
    )

    print("\n" + "="*50)
    print("STARTING WHISPER ASR TEST")
    print("="*50 + "\n")
    # 4. Run test on the first 100 samples
    for i in range(100):
        audio_file_path = dataset[i]["audio_path"]
        ground_truth_tr = dataset[i]["sentence"]
        
        audio_array, _ = librosa.load(audio_file_path, sr=16000)
        
        result = asr_pipe(
            audio_array, 
            generate_kwargs={"language": "turkish", "suppress_tokens": ""}
        )
        predicted_tr = result["text"].strip()
        
        print(f"Sample #{i+1}")
        print(f"Ground Truth (TR): {ground_truth_tr}")
        print(f"Whisper Output (TR): {predicted_tr}")
        print("-" * 30)

if __name__ == "__main__":
    main()