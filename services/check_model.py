import torch
import librosa
from datasets import load_from_disk
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer

def main():
    print("--- MODEL DIAGNOSTIC TEST ---")
    
    model_path = "./e2e_full_results/best_model"
    encoder_id = "facebook/wav2vec2-large-xlsr-53"
    decoder_id = "Helsinki-NLP/opus-mt-tr-en"

    # Yüklemeler
    extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)
    model = SpeechEncoderDecoderModel.from_pretrained(model_path).to("cuda:0")

    # --- HAYAT KURTARAN DÜZELTME (THE LIFESAVER FIX) ---
    # Hugging Face'in yüklemeyi unuttuğu LM_HEAD (Ağız) ağırlığını, 
    # modelin kendi içindeki kelime dağarcığından kopyalayarak yerine takıyoruz:
    model.decoder.lm_head.weight = model.decoder.model.decoder.embed_tokens.weight
    # ---------------------------------------------------

    # İlk veriyi çek
    dataset = load_from_disk("ready_covost_dataset")

    for _ in range(5):
    
        sample = dataset[_]

        # Sesi hazırla
        audio, _ = librosa.load(sample["audio_path"], sr=16000)
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt").to("cuda:0")

        # Çeviri üret
        with torch.no_grad():
            out = model.generate(
                inputs["input_values"], 
                max_length=50,
                decoder_start_token_id=tokenizer.pad_token_id
            )

        # Çıktıları karşılaştır
        prediction = tokenizer.decode(out[0], skip_special_tokens=True)
        
        print("\n" + "="*50)
        print("GERÇEK İNGİLİZCE ÇEVİRİ :", sample["translation"])
        print("MODELİN ÜRETTİĞİ ÇIKTI  :", f"'{prediction}'")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()