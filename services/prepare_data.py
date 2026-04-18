import os
import pandas as pd
from datasets import Dataset

# --- ENTER DIRECTORY PATHS HERE ---
# File containing Turkish texts from the downloaded Common Voice folder
cv_tsv_path = "../1774205200568-cv-corpus-25.0-2026-03-09-tr/cv-corpus-25.0-2026-03-09/tr/validated.tsv" 
# File containing English translations downloaded from Meta
covost_tsv_path = "../covost_v2.tr_en.tsv"   
# Folder containing the audio files
clips_folder = "../1774205200568-cv-corpus-25.0-2026-03-09-tr/cv-corpus-25.0-2026-03-09/tr/clips"        

print("1. Reading TSV files...")
cv_df = pd.read_csv(cv_tsv_path, sep='\t', low_memory=False)
covost_df = pd.read_csv(covost_tsv_path, sep='\t', low_memory=False)

print("2. Matching Turkish texts with English translations...")
merged_df = pd.merge(cv_df, covost_df, on='path', how='inner')

print("3. Checking if files exist on disk...")
merged_df['audio_path'] = merged_df['path'].apply(lambda x: os.path.join(clips_folder, x))
merged_df = merged_df[merged_df['audio_path'].apply(os.path.exists)]

print("4. Creating Hugging Face Dataset...")
final_df = merged_df[['audio_path', 'sentence', 'translation']]
dataset = Dataset.from_pandas(final_df, preserve_index=False)

print("\n--- PROCESS COMPLETE! ---")
dataset.save_to_disk("ready_covost_dataset")
print("Dataset successfully saved to the 'ready_covost_dataset' directory!")