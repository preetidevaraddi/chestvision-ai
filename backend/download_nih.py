import urllib.request
import os

# NIH ChestX-ray14 official Box links
# Note: Because Box dynamically generates direct download URLs, these links
# must be exported from the official Box portal (https://nihcc.app.box.com/v/ChestXray-NIHCC)
# or obtained via the official `batch_download_zips.py` provided by NIH.

NIH_LINKS = [
    # Replace these placeholders with the 12 direct .tar.gz download links from NIH
    # "https://nihcc.box.com/shared/static/...",
]

# Example placeholder for Data_entry_2017.csv
METADATA_LINK = "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.csv" 

DOWNLOAD_DIR = r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\raw\nih_official"

def download_file(url, dest_dir):
    filename = url.split('/')[-1]
    if "?" in filename:
        filename = filename.split("?")[0]
        
    filepath = os.path.join(dest_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Skipping {filename}, already downloaded.")
        return
        
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Successfully downloaded {filename}.")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print("Downloading Official NIH Metadata...")
    download_file(METADATA_LINK, DOWNLOAD_DIR)
    
    print("Downloading NIH Image Archives (~42 GB total)...")
    if not NIH_LINKS:
        print("ERROR: Please populate NIH_LINKS with the official Box direct URLs.")
    else:
        for link in NIH_LINKS:
            download_file(link, DOWNLOAD_DIR)
        
    print("Download process complete.")
