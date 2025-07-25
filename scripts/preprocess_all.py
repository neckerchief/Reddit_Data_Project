import pandas as pd
import os
from pathlib import Path
from text_preprocessing import preprocess_dataframe

raw_dir = Path("data/raw")
processed_dir = Path("data/processed")

# Ensure processed directory exists
processed_dir.mkdir(parents=True, exist_ok=True)

# Loop through CSV files in data/raw directory
for csv_file in raw_dir.glob("*.csv"):
    df = pd.read_csv(csv_file)

    # Apply processing
    df_processed = preprocess_dataframe(df, text_col1='title', text_col2='selftext')

    # Save processed DataFrame to the faile with the same name in processed_dir
    output_path = processed_dir / csv_file.name
    df_processed.to_csv(output_path, index=False)

    print(f"Processed: {csv_file.name} -> {output_path}")