"""
Shared paths, model builders, and generation defaults for end-to-end
speech translation (Wav2Vec2 encoder + Marian decoder).
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechEncoderDecoderModel,
)

SERVICES_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVICES_DIR.parent

DATASET_PATH = REPO_ROOT / "ready_covost_dataset"
E2E_OUTPUT_DIR = REPO_ROOT / "e2e_full_results"
E2E_BEST_MODEL_DIR = E2E_OUTPUT_DIR / "best_model"

ENCODER_MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
DECODER_MODEL_ID = "Helsinki-NLP/opus-mt-tr-en"

# Generation defaults (used by check + eval)
DEFAULT_GEN_MAX_LENGTH = 128
DEFAULT_NUM_BEAMS = 4


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_feature_extractor():
    return AutoFeatureExtractor.from_pretrained(ENCODER_MODEL_ID)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(DECODER_MODEL_ID)


def load_processors_for_inference(model_dir: Optional[Path] = None):
    """Prefer tokenizer + feature extractor saved with the checkpoint; else hub ids."""
    path = Path(model_dir) if model_dir is not None else E2E_BEST_MODEL_DIR
    tokenizer = (
        AutoTokenizer.from_pretrained(str(path))
        if (path / "tokenizer_config.json").is_file()
        else load_tokenizer()
    )
    extractor = (
        AutoFeatureExtractor.from_pretrained(str(path))
        if (path / "preprocessor_config.json").is_file()
        else load_feature_extractor()
    )
    return extractor, tokenizer


def configure_e2e_generation_config(
    model: SpeechEncoderDecoderModel,
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Align SpeechEncoderDecoder config with the Marian tokenizer and decoder config."""
    decoder_config = AutoConfig.from_pretrained(DECODER_MODEL_ID)
    model.config.decoder_start_token_id = decoder_config.decoder_start_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size


def build_speech_encoder_decoder_model() -> SpeechEncoderDecoderModel:
    """Fresh encoder–decoder stack with Marian-aligned generation config."""
    tokenizer = load_tokenizer()
    model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
        ENCODER_MODEL_ID,
        DECODER_MODEL_ID,
    )
    configure_e2e_generation_config(model, tokenizer)
    model.freeze_feature_encoder()
    return model


def load_trained_e2e_model(
    model_dir: Optional[Path] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> SpeechEncoderDecoderModel:
    """Load fine-tuned checkpoint without mutating decoder weights."""
    path = Path(model_dir) if model_dir is not None else E2E_BEST_MODEL_DIR
    dev = device if device is not None else default_device()
    model = SpeechEncoderDecoderModel.from_pretrained(path)
    model.eval()
    return model.to(dev)


def generation_kwargs(model: SpeechEncoderDecoderModel) -> Dict[str, Any]:
    return {
        "max_length": DEFAULT_GEN_MAX_LENGTH,
        "num_beams": DEFAULT_NUM_BEAMS,
        "early_stopping": True,
        "decoder_start_token_id": model.config.decoder_start_token_id,
    }


def seq2seq_trainer_processing_kwarg(tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    """Support both `tokenizer=` (older HF) and `processing_class=` (newer HF)."""
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if "processing_class" in sig.parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in sig.parameters:
        return {"tokenizer": tokenizer}
    return {}


def training_precision_kwargs() -> Dict[str, Any]:
    """bf16 when supported, else fp16 on CUDA; none on CPU."""
    if not torch.cuda.is_available():
        return {}
    if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return {"bf16": True}
    return {"fp16": True}


def seq2seq_training_eval_strategy_kwarg(value: str) -> Dict[str, Any]:
    """`eval_strategy` (newer HF) vs `evaluation_strategy` (older HF)."""
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        return {"eval_strategy": value}
    return {"evaluation_strategy": value}
