import torch
from transformers import (
    SpeechEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer
)

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Building E2E Model on {device}...\n")

    # 1. STANDART VE STABİL MODELE GERİ DÖNÜYORUZ
    encoder_id = "facebook/wav2vec2-large-xlsr-53"
    decoder_id = "Helsinki-NLP/opus-mt-tr-en"

    # 2. İşleyicileri Yükle
    print("Loading Feature Extractor (Wav2Vec2) and Tokenizer (Opus-MT Base)...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
    tokenizer = AutoTokenizer.from_pretrained(decoder_id)

    # 3. Modelleri Birleştir (Hack koduna gerek yok!)
    print("Stitching Encoder and Decoder together...")
    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_id, 
        decoder_id
    )

    # 4. Eğitim İçin Gerekli Ayarlar
    model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Sesi işleyen CNN katmanlarını dondurarak ekran kartı belleğini koru
    model.freeze_feature_encoder()
    model.to(device)

    # 5. Model Bilgileri
    print("\n" + "="*50)
    print("SUCCESS: End-to-End Model Built Successfully!")
    print("="*50)
    print(f"Total Parameters : {model.num_parameters():,}")
    print("Encoder          : Wav2Vec2 (Multilingual XLSR-53)")
    print("Decoder          : MarianMT (Helsinki-NLP Base)")

if __name__ == "__main__":
    main()