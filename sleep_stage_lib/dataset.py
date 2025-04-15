import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from loguru import logger
import typer

from sleep_stage_lib.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main():
    input_path: Path = RAW_DATA_DIR / "app5.zip"
    output_path: Path = PROCESSED_DATA_DIR / "app5_dataset.csv"
    temp_dir: Path = RAW_DATA_DIR / "unzipped_data"
    if not os.path.exists(output_path):
        logger.info("Unzipping the archive...")
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        # Unzip the archive
        with ZipFile(input_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

    logger.info("Processing dataset...")
    # Data collection
    data_list = []
    # Base directory after extraction
    base_dir = os.path.join(temp_dir, "Data")
    # Walk through 'Train' and 'Test'
    for set_type in ['Train', 'Test']:
        for stage in ['awake', 'nonrem', 'rem']:
            folder_path = os.path.join(base_dir, set_type, stage)
            for filename in os.listdir(folder_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(folder_path, filename)
                    df = pd.read_csv(file_path)
                    df['stage'] = stage
                    df['set'] = set_type
                    df['file'] = filename
                    data_list.append(df)

    # Combine all into one DataFrame
    full_df = pd.concat(data_list, ignore_index=True)

    # Write the DataFrame to a CSV file
    full_df.to_csv(output_path, index=False)

    logger.info("Dataset processed and saved to CSV.")
    logger.info("Cleaning up temporary files...")
    # Clean up the temporary directory
    shutil.rmtree(temp_dir)
    logger.info("Cleaning up complete.")
    logger.success("Finished processing dataset.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
