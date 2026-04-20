import random

import librosa
import torch
from datasets import load_from_disk

from e2e_common import (
    DATASET_PATH,
    E2E_BEST_MODEL_DIR,
    generation_kwargs,
    load_processors_for_inference,
    load_trained_e2e_model,
)


def main():
    print("--- MODEL DIAGNOSTIC TEST ---")

    extractor, tokenizer = load_processors_for_inference(E2E_BEST_MODEL_DIR)
    model = load_trained_e2e_model(E2E_BEST_MODEL_DIR, device="cuda:0")

    dataset = load_from_disk(str(DATASET_PATH))
    n = min(5, len(dataset))
    if len(dataset) >= n:
        indices = random.sample(range(len(dataset)), n)
    else:
        indices = list(range(len(dataset)))

    gen_args = generation_kwargs(model)

    for idx in indices:
        sample = dataset[idx]
        audio, _ = librosa.load(sample["audio_path"], sr=16000)
        inputs = extractor(audio, sampling_rate=16000, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            out = model.generate(inputs["input_values"], **gen_args)

        prediction = tokenizer.decode(out[0], skip_special_tokens=True)

        print("\n" + "=" * 50)
        print(f"SAMPLE INDEX              : {idx}")
        print("GERÇEK İNGİLİZCE ÇEVİRİ :", sample["translation"])
        print("MODELİN ÜRETTİĞİ ÇIKTI  :", f"'{prediction}'")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
