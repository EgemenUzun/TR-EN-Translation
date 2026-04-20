import torch

from e2e_common import build_speech_encoder_decoder_model, default_device


def main():
    device = default_device()
    print(f"Building E2E model on {device}...\n")

    print("Loading and stitching encoder + decoder...")
    model = build_speech_encoder_decoder_model()
    model.to(device)

    print("\n" + "=" * 50)
    print("SUCCESS: End-to-end model built successfully")
    print("=" * 50)
    print(f"Total parameters : {model.num_parameters():,}")
    print("Encoder          : Wav2Vec2 (multilingual XLSR-53)")
    print("Decoder          : MarianMT (Helsinki-NLP opus-mt-tr-en)")


if __name__ == "__main__":
    main()
