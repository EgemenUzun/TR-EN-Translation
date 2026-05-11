"""
Cascade Demo — Whisper-large-v3 ASR + opus-mt-tc-big-tr-en MT

Her örnek için 4 satır basar:
  TR (Gerçek)  : veri setindeki asıl Türkçe metin
  TR (Whisper) : Whisper'ın sesten anladığı Türkçe
  EN (Çeviri)  : MarianMT'nin ürettiği İngilizce
  EN (Gerçek)  : veri setindeki asıl İngilizce referans
"""

import argparse
import torch
import librosa
from datasets import load_from_disk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

ASR_MODEL = "openai/whisper-large-v3"
MT_MODEL  = "Helsinki-NLP/opus-mt-tc-big-tr-en"
DATASET_PATH = "ready_covost_dataset"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=10,
                   help="Gösterilecek örnek sayısı (varsayılan: 10)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"ASR    : {ASR_MODEL}")
    print(f"MT     : {MT_MODEL}\n")

    print("Modeller yükleniyor...")
    asr_pipe = pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device)

    mt_tokenizer = AutoTokenizer.from_pretrained(MT_MODEL)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL).to(device)
    mt_model.eval()

    print("Veri seti yükleniyor...")
    dataset = load_from_disk(DATASET_PATH)
    subset = dataset.select(range(args.num_samples))
    print(f"{args.num_samples} örnek seçildi.\n")

    for i, item in enumerate(subset, 1):
        audio, _ = librosa.load(item["audio_path"], sr=16000)

        # ASR
        asr_result = asr_pipe(audio, generate_kwargs={"language": "turkish", "suppress_tokens": ""})
        pred_tr = asr_result["text"].strip()

        # MT
        if pred_tr:
            inputs = mt_tokenizer(pred_tr, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = mt_model.generate(**inputs, num_beams=4, max_length=128)
            pred_en = mt_tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            pred_en = ""

        print(f"{'─'*70}  #{i}")
        print(f"  TR (Gerçek)  : {item['sentence']}")
        print(f"  TR (Whisper) : {pred_tr}")
        print(f"  EN (Çeviri)  : {pred_en}")
        print(f"  EN (Gerçek)  : {item['translation']}")

    print(f"\n{'─'*70}")
    print("Bitti.")


if __name__ == "__main__":
    main()
