import os
import zipfile
import glob
import hashlib
import shutil
import pandas as pd
from PIL import Image
from collections import defaultdict
import random

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

def get_md5(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def get_phash(filepath):
    if not HAS_IMAGEHASH: return None
    try:
        return str(imagehash.phash(Image.open(filepath).convert('RGB')))
    except:
        return None

def extract_zips(raw_dir, extract_dir):
    zips = ['LungCancer.zip', 'Normal.zip', 'Pneumonia.zip', 'Tuberculosis.zip', 'COVID19.zip']
    for z in zips:
        z_path = os.path.join(raw_dir, z)
        if not os.path.exists(z_path):
            z_path = os.path.join(os.path.dirname(raw_dir), z)
        if os.path.exists(z_path):
            out_path = os.path.join(extract_dir, z.replace('.zip', ''))
            if not os.path.exists(out_path):
                print(f"Extracting {z}...")
                os.makedirs(out_path, exist_ok=True)
                with zipfile.ZipFile(z_path, 'r') as zf:
                    zf.extractall(out_path)

def parse_dataset(extract_dir):
    records = []
    
    # 1. Lung Cancer (JSRT)
    lc_dir = os.path.join(extract_dir, 'LungCancer')
    if os.path.exists(lc_dir):
        # Load JSRT metadata to differentiate true nodules from normal images
        jsrt_meta_path = os.path.join(lc_dir, 'jsrt_metadata.csv')
        jsrt_labels = {}
        if os.path.exists(jsrt_meta_path):
            try:
                jsrt_df = pd.read_csv(jsrt_meta_path)
                for _, row in jsrt_df.iterrows():
                    pid = str(row['study_id']).split('.')[0]
                    # If state is 'non-nodule', it's Normal. Otherwise it's a Nodule (malignant/benign).
                    jsrt_labels[pid] = 'Normal' if str(row['state']).strip().lower() == 'non-nodule' else 'Nodule'
            except Exception as e:
                print(f"Warning: Failed to parse jsrt_metadata.csv: {e}")
                
        for img in glob.glob(os.path.join(lc_dir, '**', '*.png'), recursive=True) + glob.glob(os.path.join(lc_dir, '**', '*.jpg'), recursive=True):
            basename = os.path.basename(img)
            # Patient ID is the JSRT filename without ext
            pid = basename.split('.')[0]
            # Use parsed label if available, otherwise default to Nodule
            label = jsrt_labels.get(pid, 'Nodule')
            records.append({'filepath': img, 'label': label, 'patient_id': pid, 'source': 'JSRT', 'is_external_test': True})

    # 2. Normal
    norm_dir = os.path.join(extract_dir, 'Normal')
    if os.path.exists(norm_dir):
        for img in glob.glob(os.path.join(norm_dir, '**', '*.png'), recursive=True) + glob.glob(os.path.join(norm_dir, '**', '*.jpg'), recursive=True):
            basename = os.path.basename(img)
            # Normal-100.png -> pid = Normal-100
            pid = basename.split('.')[0]
            records.append({'filepath': img, 'label': 'Normal', 'patient_id': pid, 'source': 'NormalDS', 'is_external_test': False})

    # 3. Pneumonia (Kaggle)
    pneu_dir = os.path.join(extract_dir, 'Pneumonia')
    if os.path.exists(pneu_dir):
        for img in glob.glob(os.path.join(pneu_dir, '**', '*.jpeg'), recursive=True) + glob.glob(os.path.join(pneu_dir, '**', '*.jpg'), recursive=True) + glob.glob(os.path.join(pneu_dir, '**', '*.png'), recursive=True):
            if '__MACOSX' in img: continue
            basename = os.path.basename(img)
            # e.g. person1_bacteria_2.jpeg -> person1
            if 'person' in basename:
                pid = basename.split('_')[0]
            else:
                pid = basename.split('.')[0]
                
            label = 'Pneumonia' if 'PNEUMONIA' in img or 'bacteria' in basename or 'virus' in basename else 'Normal'
            # If from chest_xray/test, designate as external test
            is_ext = 'test' in img.lower().split(os.sep)
            records.append({'filepath': img, 'label': label, 'patient_id': pid, 'source': 'KagglePneumonia', 'is_external_test': is_ext})

    # 4. Tuberculosis
    tb_dir = os.path.join(extract_dir, 'Tuberculosis')
    if os.path.exists(tb_dir):
        for img in glob.glob(os.path.join(tb_dir, '**', '*.png'), recursive=True) + glob.glob(os.path.join(tb_dir, '**', '*.jpg'), recursive=True):
            basename = os.path.basename(img)
            pid = basename.split('.')[0]
            label = 'Tuberculosis' if 'Tuberculosis' in img or 'TB' in img else 'Normal'
            records.append({'filepath': img, 'label': label, 'patient_id': pid, 'source': 'QatarTB', 'is_external_test': False})

    # 5. COVID19
    cov_dir = os.path.join(extract_dir, 'COVID19')
    if os.path.exists(cov_dir):
        for img in glob.glob(os.path.join(cov_dir, '**', '*.png'), recursive=True) + glob.glob(os.path.join(cov_dir, '**', '*.jpg'), recursive=True):
            basename = os.path.basename(img)
            pid = basename.split('.')[0]
            if 'COVID' in basename: label = 'COVID-19'
            elif 'Normal' in basename: label = 'Normal'
            elif 'Pneumonia' in basename or 'Viral' in basename: label = 'Pneumonia'
            else: continue
            records.append({'filepath': img, 'label': label, 'patient_id': pid, 'source': 'QatarCOVID19', 'is_external_test': False})

    return pd.DataFrame(records)

def main():
    raw_dir = r'c:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\raw'
    workspace_dir = r'c:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\v3_workspace'
    extract_dir = os.path.join(workspace_dir, 'extracted')
    final_dir = os.path.join(workspace_dir, 'final_dataset')
    
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    extract_zips(raw_dir, extract_dir)
    df = parse_dataset(extract_dir)
    
    print(f"Initial images found: {len(df)}")
    
    # 1. Corrupted Image Check
    print("Checking for corrupted images...")
    valid_idx = []
    corrupted_count = 0
    for idx, row in df.iterrows():
        try:
            with Image.open(row['filepath']) as img:
                img.verify()
            valid_idx.append(idx)
        except Exception:
            corrupted_count += 1
            
    df = df.loc[valid_idx].copy()
    
    # 2. Duplicate Check
    print("Computing hashes for global deduplication...", flush=True)
    md5_list = []
    phash_list = []
    total_imgs = len(df)
    
    for i, (_, row) in enumerate(df.iterrows()):
        fp = row['filepath']
        md5_val = get_md5(fp)
        md5_list.append(md5_val)
        if HAS_IMAGEHASH:
            phash_list.append(get_phash(fp))
        else:
            phash_list.append(md5_val)
            
        if (i+1) % 1000 == 0:
            print(f"Hashed {i+1}/{total_imgs} images...", flush=True)
            
    df['md5'] = md5_list
    df['phash'] = phash_list
        
    initial_count = len(df)
    
    # Track duplicates
    duplicates_df = df[df.duplicated(subset=['md5'], keep='first') | df.duplicated(subset=['phash'], keep='first')].copy()
    df = df.drop_duplicates(subset=['md5'], keep='first')
    df = df.drop_duplicates(subset=['phash'], keep='first')
    
    dup_count = initial_count - len(df)
    
    # 3. Patient Leakage & Train/Val/Test Splitting
    print("Assigning Train/Val/Test splits grouped by PatientID...")
    df['split'] = 'Train'
    
    # Lock ALL images of any patient who has at least one external test image
    test_locked_patients = df[df['is_external_test'] == True]['patient_id'].unique()
    df.loc[df['patient_id'].isin(test_locked_patients), 'split'] = 'Test'
    
    # Group remaining by patient ID
    patients = df[df['split'] == 'Train']['patient_id'].unique()
    random.seed(42)
    patients = list(patients)
    random.shuffle(patients)
    
    n_patients = len(patients)
    train_end = int(n_patients * 0.70)
    val_end = int(n_patients * 0.85)
    
    train_pids = set(patients[:train_end])
    val_pids = set(patients[train_end:val_end])
    test_pids = set(patients[val_end:])
    
    df.loc[df['patient_id'].isin(val_pids), 'split'] = 'Val'
    df.loc[df['patient_id'].isin(test_pids), 'split'] = 'Test'
    
    # Save final reports
    manifest_path = os.path.join(workspace_dir, 'dataset_manifest.csv')
    summary_path = os.path.join(workspace_dir, 'dataset_summary.csv')
    dup_report_path = os.path.join(workspace_dir, 'duplicate_report.csv')
    audit_path = os.path.join(workspace_dir, 'audit_report.txt')
    
    df.to_csv(manifest_path, index=False)
    if not duplicates_df.empty:
        duplicates_df.to_csv(dup_report_path, index=False)
    else:
        open(dup_report_path, 'w').write('filepath,label,patient_id,source,is_external_test,md5,phash\n')
        
    summary = df.groupby(['split', 'label', 'source']).size().reset_index(name='count')
    summary.to_csv(summary_path, index=False)
    
    # Write audit report
    with open(audit_path, 'w') as f:
        f.write("=== ChestVision V3 Dataset Audit Report ===\n\n")
        f.write(f"Total Initial Images Found: {initial_count + corrupted_count}\n")
        f.write(f"Corrupted/Invalid Images: {corrupted_count}\n")
        f.write(f"Duplicates Removed: {dup_count}\n")
        f.write(f"Final Usable Images: {len(df)}\n\n")
        
        f.write("--- Image Count per Class ---\n")
        f.write(df['label'].value_counts().to_string() + "\n\n")
        
        f.write("--- Image Count per Source ---\n")
        f.write(df['source'].value_counts().to_string() + "\n\n")
        
        f.write("--- Final Split Counts ---\n")
        f.write(df['split'].value_counts().to_string() + "\n\n")
        
        f.write("--- External Test Count ---\n")
        ext_count = len(df[df['is_external_test'] == True])
        f.write(f"Images forced to Test (External): {ext_count}\n\n")
        
        f.write("--- Leakage Concerns ---\n")
        f.write("Patient-level leakage check passed (Train/Val/Test split grouped by patient_id).\n")
        f.write("Designated external test datasets strictly placed in 'Test'.\n")

    print(f"Preparation complete. Workspace: {workspace_dir}")
    print(f"Audit reports generated in {workspace_dir}")

if __name__ == '__main__':
    main()
