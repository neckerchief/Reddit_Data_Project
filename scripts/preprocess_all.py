import pandas as pd
import os
from pathlib import Path
from text_preprocessing import preprocess_dataframe
from feature_engineering import create_all_features

# Define paths
raw_dir = Path("data/raw")
processed_dir = Path("data/processed")
master_file = raw_dir / "reddit_posts_master.csv"
output_file = processed_dir / "reddit_posts_master_processed.parquet"

# Ensure processed directory exists
processed_dir.mkdir(parents=True, exist_ok=True)

# Check if master file exists
if not master_file.exists():
    print(f"ERROR: Master file not found: {master_file}")
    print("Make sure you've run the scraper first!")
    exit(1)

print(f"Processing master file: {master_file}")
print("Loading data...")

# Read master file
df = pd.read_csv(master_file)
print(f"Loaded {len(df)} rows")

# Apply text preprocessing
print("Applying text preprocessing...")
df_processed = preprocess_dataframe(df, text_col1='title', text_col2='selftext')

# Apply feature engineering
print("Creating features...")
df_processed_further = create_all_features(df_processed)

# Save processed DataFrame
print(f"Saving to: {output_file}")
df_processed_further.to_parquet(output_file, index=False)
print("=== PREPROCESSING COMPLETE ===")
print(f"Input rows: {len(df)}")
print(f"Output rows: {len(df_processed_further)}")
print(f"Output columns: {len(df_processed_further.columns)}")
print(f"Processed file: {output_file}")