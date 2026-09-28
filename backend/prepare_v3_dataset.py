import os
import glob
import hashlib
import shutil
import pandas as pd
from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("Warning: imagehash not installed. Perceptual hashing will be skipped. Run 'pip install imagehash' for full deduplication.")

def get_md5(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def get_phash(filepath):
    if not HAS_IMAGEHASH:
        return None
    try:
        img = Image.open(filepath)
        return str(imagehash.phash(img))
    except:
        return None

def setup_dirs():
    dirs = ['dataset/v3_downloads', 'dataset/v3_images/Train', 'dataset/v3_images/Test']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def parse_nih_metadata():
    csv_path = 'dataset/v3_downloads/nih/Data_entry_2017.csv'
    if not os.path.exists(csv_path):
        print(f"NIH metadata not found at {csv_path}")
        return pd.DataFrame()
        
    df = pd.read_csv(csv_path)
    # STRICT SINGLE-LABEL FILTERING
    # Reject any row where Finding Labels contains a pipe '|'
    single_label_df = df[~df['Finding Labels'].str.contains('\|', regex=True)].copy()
    
    # We only want Pneumonia, Nodule, and No Finding
    valid_classes = ['Pneumonia', 'Nodule', 'No Finding']
    filtered_df = single_label_df[single_label_df['Finding Labels'].isin(valid_classes)].copy()
    
    # Map to our classes
    label_map = {'No Finding': 'Normal', 'Pneumonia': 'Pneumonia', 'Nodule': 'Nodule'}
    filtered_df['TargetClass'] = filtered_df['Finding Labels'].map(label_map)
    filtered_df['PatientID'] = filtered_df['Patient ID']
    filtered_df['Source'] = 'NIH'
    filtered_df['ImageName'] = filtered_df['Image Index']
    
    return filtered_df[['ImageName', 'TargetClass', 'PatientID', 'Source']]

def parse_cohen_metadata():
    csv_path = 'dataset/v3_downloads/cohen/metadata.csv'
    if not os.path.exists(csv_path):
        print(f"Cohen metadata not found at {csv_path}")
        return pd.DataFrame()
        
    df = pd.read_csv(csv_path)
    # Filter for X-ray only
    df = df[df['modality'] == 'X-ray'].copy()
    # Filter for Frontal views (PA / AP), exclude AP Supine
    df = df[df['view'].isin(['PA', 'AP'])].copy()
    # Positive COVID-19 only
    df = df[df['finding'] == 'Pneumonia/Viral/COVID-19'].copy()
    
    df['TargetClass'] = 'COVID-19'
    df['Source'] = 'Cohen'
    df['PatientID'] = df['patientid']
    df['ImageName'] = df['filename']
    
    return df[['ImageName', 'TargetClass', 'PatientID', 'Source']]

def global_deduplication_and_split(df_list, images_base_dir):
    print("Running global MD5 and pHash deduplication...")
    master_df = pd.concat(df_list, ignore_index=True)
    
    seen_md5 = set()
    seen_phash = set()
    clean_records = []
    
    for _, row in master_df.iterrows():
        img_path = os.path.join(images_base_dir, row['Source'], row['ImageName'])
        if not os.path.exists(img_path):
            continue
            
        md5_val = get_md5(img_path)
        if md5_val in seen_md5:
            continue
        seen_md5.add(md5_val)
        
        if HAS_IMAGEHASH:
            phash_val = get_phash(img_path)
            if phash_val and phash_val in seen_phash:
                continue
            if phash_val:
                seen_phash.add(phash_val)
                
        clean_records.append(row)
        
    clean_df = pd.DataFrame(clean_records)
    print(f"Total candidate images: {len(master_df)}")
    print(f"Clean usable images after deduplication: {len(clean_df)}")
    return clean_df

if __name__ == '__main__':
    setup_dirs()
    print("Dataset Preparation Script (Strict Single-Label & Deduplication) initialized.")
    print("Note: Actual Kaggle API downloading logic is intentionally commented out to prevent massive unauthorized downloads.")
    # Implement actual API calls and file moving logic here once authorized.
