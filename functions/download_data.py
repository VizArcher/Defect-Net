"""
Contains functionality for downloading data from Kaggle into Google Colab
"""
import os
import zipfile
import subprocess
import shutil  # Make sure shutil is imported
from google.colab import drive
from pathlib import Path

def download_data(dataset_name):
    """
    Download data from Kaggle using credentials stored in Google Drive.

    Args:
        dataset_name: str, name of the Kaggle dataset (e.g., "username/dataset")

    Returns:
        None
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "3D_Defect"

    # If the image folder doesn't exist, create it
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_name, "-p", str(image_path)], check=True)

    # Unzip the downloaded dataset
    zip_file_path = str(image_path / (dataset_name.split('/')[-1] + '.zip'))
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(image_path)

    # Optionally, remove the zip file after extraction
    os.remove(zip_file_path)

    print(f"[INFO] Data download and extraction complete.")
