import pandas as pd
import os
from pathlib import Path
from text_preprocessing import preprocess_dataframe
from feature_engineering import create_all_features

raw_dir = Path("data/raw")
processed_dir = Path("data/processed")

# Debug: Check if the directory exists and what's in it
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for CSV files in: {raw_dir.resolve()}")
print(f"Directory exists: {raw_dir.exists()}")

if raw_dir.exists():
    all_files = list(raw_dir.glob("*"))
    print(f"All files in directory: {all_files}")
    csv_files = list(raw_dir.glob("*.csv"))
    print(f"CSV files found: {csv_files}")
else:
    print("Directory does not exist!")

# Ensure processed directory exists
processed_dir.mkdir(parents=True, exist_ok=True)

# Loop through CSV files in data/raw directory
for csv_file in raw_dir.glob("*.csv"):
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file)

    # Apply text preprocessing
    df_processed = preprocess_dataframe(df, text_col1='title', text_col2='selftext')

    # Apply feature engineering
    df_processed_further = create_all_features(df_processed)

    # Save processed DataFrame to the processed_dir with the same name
    output_path = processed_dir / csv_file.name
    df_processed_further.to_csv(output_path, index=False)

    print(f"Processed: {csv_file.name} -> {output_path}")