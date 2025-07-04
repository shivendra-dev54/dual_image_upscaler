import os
import shutil
import zipfile
import urllib.request

def download_data():
    data_dir = os.path.join(os.getcwd(), 'data')
    zip_filename = 'DIV2K_train_HR.zip'
    zip_url = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'

    # Check if data directory exists
    if os.path.exists(data_dir):
        response = input("The 'data' directory already exists. Do you want to re-download the dataset? (y/n): ").strip().lower()
        if response != 'y':
            print("Download skipped.")
            return
        else:
            print("Removing existing 'data' directory...")
            shutil.rmtree(data_dir)

    # Create necessary directories
    os.makedirs(os.path.join(data_dir, 'train_hr'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'train_lr'), exist_ok=True)

    # Download the zip file
    print(f"Downloading dataset from {zip_url}...")
    urllib.request.urlretrieve(zip_url, zip_filename)
    print("Download complete.")

    # Extract the zip file
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall()
    print("Extraction complete.")

    # Move extracted images
    extracted_folder = 'DIV2K_train_HR'
    print("Organizing dataset...")
    for file in os.listdir(extracted_folder):
        shutil.move(os.path.join(extracted_folder, file), os.path.join(data_dir, 'train_hr', file))

    # Cleanup
    shutil.rmtree(extracted_folder)
    os.remove(zip_filename)
    print("Dataset is ready in the 'data/' directory.")

