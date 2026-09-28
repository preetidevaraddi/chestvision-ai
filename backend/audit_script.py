import os
import json
import hashlib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from PIL import Image

def get_hash(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    report = {}
    
    # 1. Exact training dataset
    df_path = 'dataset/labels_cleaned.csv'
    if not os.path.exists(df_path):
        df_path = 'dataset/labels.csv'
    
    df = pd.read_csv(df_path)
    report['total_images'] = len(df)
    
    class_counts = df['label'].value_counts().to_dict()
    report['class_counts'] = class_counts
    
    # Determine sources and dimensions
    sources = {}
    dimensions = set()
    formats = set()
    patient_ids_available = 'patientid' in df.columns
    
    for _, row in df.iterrows():
        label = row['label']
        if label not in sources:
            sources[label] = set()
            
        img_path = os.path.join('dataset/images', row['filename'])
        if not os.path.exists(img_path):
            img_path = os.path.join('dataset/real_images', row['filename'])
            
        if os.path.exists(img_path):
            if 'source' in row:
                sources[label].add(row['source'])
            elif 'nih' in row['filename'].lower() or '000' in row['filename']: # heuristic for NIH vs JSRT vs RSNA
                pass # Try to figure out from filename or folder if source column is missing
                
            try:
                with Image.open(img_path) as img:
                    dimensions.add(img.size)
                    formats.add(img.format)
            except:
                pass
                
    report['dimensions'] = [str(d) for d in list(dimensions)[:10]]
    report['formats'] = list(formats)
    report['patient_ids_available'] = patient_ids_available
    
    # Check if there is a 'source' column
    if 'source' in df.columns:
        report['sources'] = df.groupby('label')['source'].apply(lambda x: list(x.unique())).to_dict()
    else:
        report['sources'] = "Source column missing, inferring from filenames/directories..."

    # 2. Train/validation/test split
    # Imitate split logic from final_audit.py
    min_count = df['label'].value_counts().min()
    balanced_dfs = [df[df['label'] == cls].head(min_count) for cls in df['label'].unique()]
    bdf = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if patient_ids_available:
        unique_patients = bdf['patientid'].unique()
        train_p, temp_p = train_test_split(unique_patients, test_size=0.3, random_state=42)
        val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)
        
        train_df = bdf[bdf['patientid'].isin(train_p)]
        val_df = bdf[bdf['patientid'].isin(val_p)]
        test_df = bdf[bdf['patientid'].isin(test_p)]
        
        report['split'] = {
            'train_images': len(train_df),
            'val_images': len(val_df),
            'test_images': len(test_df),
            'train_patients': len(train_p),
            'val_patients': len(val_p),
            'test_patients': len(test_p),
            'patient_overlap_train_val': len(set(train_p).intersection(val_p)),
            'patient_overlap_train_test': len(set(train_p).intersection(test_p)),
            'patient_overlap_val_test': len(set(val_p).intersection(test_p))
        }
    
    # Duplicates across splits based on hashes
    train_hashes = set()
    val_hashes = set()
    test_hashes = set()
    
    hashes = {}
    duplicates = 0
    cross_class_dupes = 0
    
    for df_split, hash_set, name in [(train_df, train_hashes, 'train'), (val_df, val_hashes, 'val'), (test_df, test_hashes, 'test')]:
        for _, row in df_split.iterrows():
            img_path = os.path.join('dataset/images', row['filename'])
            if not os.path.exists(img_path):
                img_path = os.path.join('dataset/real_images', row['filename'])
            if not os.path.exists(img_path):
                continue
            h = get_hash(img_path)
            hash_set.add(h)
            
            if h in hashes:
                duplicates += 1
                if hashes[h] != row['label']:
                    cross_class_dupes += 1
            hashes[h] = row['label']
            
    report['image_overlap_train_val'] = len(train_hashes.intersection(val_hashes))
    report['image_overlap_train_test'] = len(train_hashes.intersection(test_hashes))
    report['image_overlap_val_test'] = len(val_hashes.intersection(test_hashes))
    report['total_exact_duplicates'] = duplicates
    report['cross_class_duplicates'] = cross_class_dupes
    
    # 5. Current model configuration
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    try:
        from app.ml.model import ChestVisionModel
        
        model_path = "models_store/new_model_real/chestvision_model.pt"
        labels_path = "models_store/new_model_real/labels.json"
        
        with open(labels_path, "r") as f:
            labels = json.load(f)
            
        model = ChestVisionModel(num_classes=len(labels), pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        
        trainable = 0
        frozen = 0
        for name, param in model.backbone.named_parameters():
            if "denseblock4" in name or "norm5" in name or "classifier" in name:
                trainable += param.numel()
            else:
                frozen += param.numel()
                
        report['model'] = {
            'architecture': 'DenseNet121',
            'trainable_params': trainable,
            'frozen_params': frozen,
            'labels': labels
        }
    except Exception as e:
        report['model'] = str(e)
        
    with open('audit_results.json', 'w') as f:
        json.dump(report, f, indent=4)
        
if __name__ == '__main__':
    main()
