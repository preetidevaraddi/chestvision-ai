import os
import zipfile
import glob
import pandas as pd
import json
from PIL import Image

def extract_archives():
    raw_dir = "dataset/raw"
    extracted_dir = "dataset/extracted"
    os.makedirs(extracted_dir, exist_ok=True)
    
    archives = glob.glob(f"{raw_dir}/*.zip")
    print(f"Found {len(archives)} archives in {raw_dir}.")
    
    if len(archives) == 0:
        print("ERROR: No ZIP archives found. Please ensure the archives are placed in dataset/raw/.")
        return False
        
    for archive in archives:
        print(f"Extracting {archive}...")
        try:
            with zipfile.ZipFile(archive, 'r') as zip_ref:
                folder_name = os.path.basename(archive).replace('.zip', '')
                target_path = os.path.join(extracted_dir, folder_name)
                os.makedirs(target_path, exist_ok=True)
                zip_ref.extractall(target_path)
            print(f"  -> Successfully extracted to {target_path}")
        except zipfile.BadZipFile:
            print(f"  [ERROR] {archive} is corrupted or incomplete. Skipping.")
    return True

def audit_datasets():
    extracted_dir = "dataset/extracted"
    audit_records = []
    
    for root, dirs, files in os.walk(extracted_dir):
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            continue
            
        sample_img_path = os.path.join(root, image_files[0])
        try:
            with Image.open(sample_img_path) as img:
                dimensions = f"{img.width}x{img.height}"
        except:
            dimensions = "Unknown"
            
        dataset_name = os.path.basename(os.path.dirname(root))
        if dataset_name == "extracted":
            dataset_name = os.path.basename(root)
            
        inferred_class = os.path.basename(root)
        metadata_files = [f for f in os.listdir(os.path.dirname(root)) if f.lower().endswith('.csv') or f.lower().endswith('.json')]
        has_patient_ids = "Yes" if "metadata.csv" in metadata_files or "COVID" in dataset_name else "No"
        
        audit_records.append({
            "Dataset Path": root,
            "Source Folder": dataset_name,
            "Number of Images": len(image_files),
            "Formats": list(set([os.path.splitext(f)[1].lower() for f in image_files])),
            "Inferred Label": inferred_class,
            "Metadata Files": metadata_files if metadata_files else "None",
            "Patient IDs Available": has_patient_ids,
            "Sample Dimensions": dimensions
        })
        
    df_audit = pd.DataFrame(audit_records)
    print("\n=== DATASET AUDIT REPORT ===")
    print(df_audit.to_string())
    
    df_audit.to_csv("dataset_summary.csv", index=False)
    print("\nAudit saved to dataset_summary.csv")
    print("Ready for deduplication and strict preparation (Phase 5-11).")

if __name__ == "__main__":
    print("Phase 3-4: Extracting Archives and Auditing Structure")
    if extract_archives():
        audit_datasets()
