import os
import hashlib
import pandas as pd

def get_hash(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def clean_dataset():
    df = pd.read_csv('dataset/labels.csv')
    print(f"Original dataset size: {len(df)}")
    
    hash_to_files = {}
    file_to_label = dict(zip(df['filename'], df['label']))
    
    for _, row in df.iterrows():
        filename = row['filename']
        img_path = os.path.join('dataset/images', filename)
        if not os.path.exists(img_path):
            img_path = os.path.join('dataset/real_images', filename)
            if not os.path.exists(img_path):
                print(f"File not found: {filename}")
                continue
        
        h = get_hash(img_path)
        if h not in hash_to_files:
            hash_to_files[h] = []
        hash_to_files[h].append(filename)

    files_to_remove = set()
    
    for h, files in hash_to_files.items():
        if len(files) > 1:
            labels = {file_to_label[f] for f in files}
            if len(labels) > 1:
                # Cross-class contamination: drop all to be safe
                print(f"Removing ALL cross-class duplicates for hash {h}: {files} (Labels: {labels})")
                files_to_remove.update(files)
            else:
                # Same class duplication: keep the first, drop the rest
                print(f"Removing intra-class duplicates for hash {h}: keeping {files[0]}, dropping {files[1:]}")
                files_to_remove.update(files[1:])

    cleaned_df = df[~df['filename'].isin(files_to_remove)]
    print(f"\nRemoved {len(files_to_remove)} duplicate files.")
    print(f"Cleaned dataset size: {len(cleaned_df)}")
    
    cleaned_df.to_csv('dataset/labels_cleaned.csv', index=False)
    print("\nRemaining unique images per class:")
    print(cleaned_df['label'].value_counts())

if __name__ == '__main__':
    clean_dataset()
