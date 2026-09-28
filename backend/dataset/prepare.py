import os
import zipfile
import shutil
import random
import pandas as pd
from PIL import Image

def main():
    zips_dir = 'dataset/zips'
    extracted_dir = 'dataset/extracted'
    final_dir = 'dataset/images'
    
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # Extract all zips
    zips = [f for f in os.listdir(zips_dir) if f.endswith('.zip')]
    for z in zips:
        print(f"Extracting {z}...")
        with zipfile.ZipFile(os.path.join(zips_dir, z), 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
            
    print("Preparing final dataset...")
    
    classes = {
        'Normal': [],
        'Pneumonia': [],
        'Tuberculosis': [],
        'COVID19': [],
        'Nodule': []
    }
    
    # Traverse extracted directory to find images
    for root, dirs, files in os.walk(extracted_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, f)
                lower_path = path.lower()
                
                # Simple logic based on folder structure
                if 'normal' in lower_path:
                    classes['Normal'].append(path)
                elif 'pneumonia' in lower_path:
                    classes['Pneumonia'].append(path)
                elif 'tuberculosis' in lower_path:
                    classes['Tuberculosis'].append(path)
                elif 'covid' in lower_path:
                    classes['COVID19'].append(path)
                elif 'lungcancer' in lower_path or 'jsrt' in lower_path:
                    classes['Nodule'].append(path)
                    
    # Cap at 1500 images per class
    target_count = 1500
    
    records = []
    
    for cls, paths in classes.items():
        print(f"{cls}: found {len(paths)} images")
        
        # If we have less than target_count, we duplicate images to reach target_count
        # This is primarily for Nodule (JSRT) since NIH is missing.
        if len(paths) == 0:
            print(f"Warning: No images found for {cls}!")
            continue
            
        if len(paths) > target_count:
            random.seed(42)
            selected = random.sample(paths, target_count)
        else:
            # Duplicate to reach target_count
            selected = paths.copy()
            while len(selected) < target_count:
                selected.extend(paths[:target_count - len(selected)])
                
        for i, src_path in enumerate(selected):
            ext = os.path.splitext(src_path)[1]
            new_filename = f"{cls}_{i:04d}{ext}"
            dst_path = os.path.join(final_dir, new_filename)
            
            # Since we are duplicating, we might overwrite if we don't append a unique index.
            # i ensures uniqueness even if src_path is duplicated.
            shutil.copy(src_path, dst_path)
            
            # For multi-label CSV format expected by train.py
            # format: filename, label_1, label_2
            records.append({
                'filename': new_filename,
                'Normal': 1 if cls == 'Normal' else 0,
                'Pneumonia': 1 if cls == 'Pneumonia' else 0,
                'Tuberculosis': 1 if cls == 'Tuberculosis' else 0,
                'COVID-19': 1 if cls == 'COVID19' else 0,
                'Nodule': 1 if cls == 'Nodule' else 0
            })
            
    df = pd.DataFrame(records)
    df.to_csv('dataset/labels.csv', index=False)
    print(f"Created dataset/labels.csv with {len(df)} rows.")

if __name__ == '__main__':
    main()
