"""
E2E Speech Translation - Improved Evaluation Script (v2)

Key improvements over evaluate_e2e.py:
  - Proper generate params: eos_token_id, no_repeat_ngram_size, repetition_penalty
  - Configurable sample count (default 500, pass --all for full dataset)
  - Per-sample chrF scores saved to CSV for error analysis
  - Worst / best translation examples printed for qualitative inspection
  - Supports comparing v1 model vs v2 model side-by-side (--model-path arg)
  - Results saved to JSON for later comparison with cascade numbers
"""

import argparse
import json
import csv
import torch
import librosa
from pathlib import Path
from datasets import load_from_disk
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
import evaluate as evaluate_lib
from tqdm import tqdm


ENCODER_ID = "facebook/wav2vec2-large-xlsr-53"
DECODER_ID  = "Helsinki-NLP/opus-mt-tr-en"
DATASET_PATH = "ready_covost_dataset"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate E2E Speech Translation model")
    p.add_argument("--model-path", default="./e2e_v2_results/best_model",
                   help="Path to the trained model directory")
    p.add_argument("--num-samples", type=int, default=500,
                   help="Number of samples to evaluate (0 = full dataset)")
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--output-dir", default="./eval_results_v2",
                   help="Directory to save JSON + CSV results")
    p.add_argument("--top-n", type=int, default=5,
                   help="Number of best/worst examples to print")
    return p.parse_args()


def load_model(model_path: str, device: str):
    extractor = AutoFeatureExtractor.from_pretrained(ENCODER_ID)
    tokenizer = AutoTokenizer.from_pretrained(DECODER_ID)
    model = SpeechEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()
    return model, extractor, tokenizer


def translate(model, extractor, tokenizer, audio_path: str, device: str,
              num_beams: int, max_length: int) -> str:
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt").to(device)

    with torch.no_grad():
        ids = model.generate(
            inputs["input_values"],
            max_length=max_length,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def print_examples(rows: list, title: str, n: int):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    for i, r in enumerate(rows[:n], 1):
        print(f"\n[{i}] chrF: {r['chrf']:.2f}")
        print(f"  REF : {r['reference']}")
        print(f"  PRED: {r['prediction']}")


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device      : {device}")
    print(f"Model path  : {args.model_path}")

    print("\n1. Loading model...")
    model, extractor, tokenizer = load_model(args.model_path, device)

    print("2. Loading dataset...")
    dataset = load_from_disk(DATASET_PATH)
    if args.num_samples and args.num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(args.num_samples))
    print(f"   Evaluating on {len(dataset):,} samples")

    bleu_metric = evaluate_lib.load("sacrebleu")
    chrf_metric  = evaluate_lib.load("chrf")
    # Per-sample chrF for ranking
    chrf_per_sample = evaluate_lib.load("chrf")

    predictions, references = [], []
    rows = []

    print("\n3. Running inference...")
    for item in tqdm(dataset, desc="Translating"):
        pred = translate(
            model, extractor, tokenizer,
            item["audio_path"], device,
            args.num_beams, args.max_length,
        )
        ref = item["translation"]

        predictions.append(pred)
        references.append([ref])

        per_chrf = chrf_per_sample.compute(
            predictions=[pred], references=[[ref]]
        )["score"]
        rows.append({"reference": ref, "prediction": pred, "chrf": per_chrf})

    print("\n4. Computing corpus-level scores...")
    bleu  = bleu_metric.compute(predictions=predictions, references=references)
    chrf  = chrf_metric.compute(predictions=predictions, references=references)

    print("\n" + "=" * 55)
    print("  E2E v2 EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Samples  : {len(dataset):,}")
    print(f"  BLEU     : {bleu['score']:.2f}  (higher is better)")
    print(f"  chrF     : {chrf['score']:.2f}  (higher is better)")
    print("=" * 55)

    # Qualitative inspection
    sorted_rows = sorted(rows, key=lambda r: r["chrf"])
    print_examples(sorted_rows, f"WORST {args.top_n} TRANSLATIONS", args.top_n)
    print_examples(list(reversed(sorted_rows)), f"BEST {args.top_n} TRANSLATIONS", args.top_n)

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model_path": args.model_path,
        "num_samples": len(dataset),
        "bleu": round(bleu["score"], 4),
        "chrf": round(chrf["score"], 4),
        "num_beams": args.num_beams,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = out_dir / "per_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["reference", "prediction", "chrf"])
        writer.writeheader()
        writer.writerows(sorted_rows)

    print(f"\nResults saved to {out_dir}/")
    print(f"  summary.json  — corpus scores")
    print(f"  per_sample.csv — all {len(rows):,} rows sorted by chrF (worst first)")


if __name__ == "__main__":
    main()
