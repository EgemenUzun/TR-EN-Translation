# P14 — Turkish ↔ English Speech Translation: Cascade vs. End-to-End

## Project Overview
This project focuses on building and comparing Speech-to-Text Translation (ST) systems that translate Turkish audio directly into English text. 

The primary goal is to compare two distinct architectures:
1.  Cascade System (ASR + MT): Transcribes audio to Turkish text using an Automatic Speech Recognition (ASR) model, and then translates the text to English using a Machine Translation (MT) model.
2.  End-to-End System (E2E): Translates Turkish audio directly into English text without generating intermediate Turkish text. (Currently in progress)

## Dataset
The project utilizes the CoVoST2 dataset for Turkish-to-English translations, heavily relying on the Mozilla Common Voice (Turkish) corpus for the source audio and Turkish transcriptions.

## Installation

1. Create a Conda environment:
conda create -n st_project python=3.10 -y
conda activate st_project

2. Install the required PyTorch packages (for CUDA 11.8 support):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. Install the rest of the project dependencies:
pip install -r requirements.txt

## Project Structure & Usage

Follow the steps below to run the Cascade System pipeline:

### Step 1: Data Preparation
Merges the downloaded Common Voice .tsv files with CoVoST2 translations and verifies the existence of audio clips.

Command: python prepare_data.py
Output: Generates a Hugging Face dataset saved in the 'ready_covost_dataset' directory.

### Step 2: Test ASR Model (Whisper)
Tests the first component of the cascade system. Evaluates how well openai/whisper-small transcribes the Turkish audio into Turkish text.

Command: python whisper_asr_test.py

### Step 3: Test Full Cascade System (Whisper + Helsinki-NLP)
Runs the end-to-end cascade pipeline on a few samples. Transcribes the audio using Whisper and translates the resulting text to English using Helsinki-NLP/opus-mt-tr-en.

Command: python cascade_full_test.py

### Step 4: Evaluate the Cascade System
Evaluates the entire prepared dataset using standard academic metrics. Calculates WER (Word Error Rate) for the ASR stage, and BLEU & chrF scores for the final English translation.

Command: python evaluate_cascade_full.py

## Metrics & Evaluation
- WER (Word Error Rate): Used to measure the accuracy of the ASR model. Lower is better.
- BLEU & chrF: Used to measure the translation quality against ground-truth CoVoST2 English translations. Higher is better.

## Future Work
- Implementation of the End-to-End (E2E) ST model using wav2vec2-encoder + Transformer-decoder.
- Ablation studies (Beam search penalty, length penalty).
- Error analysis on proper nouns and long sentences.