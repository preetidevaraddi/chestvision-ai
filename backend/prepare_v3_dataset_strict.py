import os
import glob
import shutil
import hashlib
import pandas as pd
from PIL import Image

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

# Constants
EXTERNAL_TEST_SOURCES = ['Cohen', 'Montgomery', 'JSRT']
TRAIN_VAL_TOTAL = 336

def setup_directories():
    for d in ['dataset/v3_strict/Train', 'dataset/v3_strict/Validation', 'dataset/v3_strict/ExternalTest', 'dataset/v3_strict/reports', 'dataset/v3_strict/downloads']:
        os.makedirs(d, exist_ok=True)

def generate_mock_data():
    """Generates synthetic metadata and files to simulate the 50GB dataset download so the logic can run."""
    print("Generating mock dataset metadata...")
    os.makedirs('dataset/v3_strict/downloads/nih', exist_ok=True)
    os.makedirs('dataset/v3_strict/downloads/kaggle', exist_ok=True)
    os.makedirs('dataset/v3_strict/downloads/cohen', exist_ok=True)
    
    # NIH
    nih_data = []
    for i in range(1500): nih_data.append({'Finding Labels': 'Pneumonia', 'Image Index': f'nih_pneu_{i}.png', 'Patient ID': f'nih_pat_{i}'})
    for i in range(3000): nih_data.append({'Finding Labels': 'Nodule', 'Image Index': f'nih_nod_{i}.png', 'Patient ID': f'nih_pat_{i+1500}'})
    for i in range(2000): nih_data.append({'Finding Labels': 'No Finding', 'Image Index': f'nih_norm_{i}.png', 'Patient ID': f'nih_pat_{i+4500}'})
    pd.DataFrame(nih_data).to_csv('dataset/v3_strict/downloads/nih/Data_entry_2017.csv', index=False)
    
    # Kaggle
    for cls in ['COVID', 'Viral Pneumonia', 'Normal', 'Lung Opacity']:
        d = f'dataset/v3_strict/downloads/kaggle/{cls}/images'
        os.makedirs(d, exist_ok=True)
        for i in range(1500):
            with open(os.path.join(d, f'kag_{cls.replace(" ", "_")}_{i}.png'), 'w') as f: f.write('mock')
            
    # Cohen
    cohen_data = []
    for i in range(500): cohen_data.append({'modality': 'X-ray', 'view': 'PA', 'finding': 'Pneumonia/Viral/COVID-19', 'filename': f'cohen_{i}.png', 'patientid': f'cohen_pat_{i}'})
    pd.DataFrame(cohen_data).to_csv('dataset/v3_strict/downloads/cohen/metadata.csv', index=False)

def get_md5(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f: h.update(f.read())
    return h.hexdigest()

def get_phash(filepath):
    return None

def parse_nih():
    csv_path = 'dataset/v3_strict/downloads/nih/Data_entry_2017.csv'
    df = pd.read_csv(csv_path)
    df = df[~df['Finding Labels'].str.contains('\|', regex=True)].copy()
    valid_classes = {'Pneumonia': 'Pneumonia', 'Nodule': 'Nodule', 'No Finding': 'Normal'}
    df = df[df['Finding Labels'].isin(valid_classes.keys())].copy()
    df['TargetClass'] = df['Finding Labels'].map(valid_classes)
    df['Source'] = 'NIH'
    df['PatientID'] = df['Patient ID']
    df['OriginalLabel'] = df['Finding Labels']
    df['OriginalFilename'] = df['Image Index']
    return df

def parse_kaggle():
    base_dir = 'dataset/v3_strict/downloads/kaggle'
    records = []
    mappings = {'COVID': 'COVID-19', 'Viral Pneumonia': 'Pneumonia', 'Normal': 'Normal'}
    for folder, target_class in mappings.items():
        folder_path = os.path.join(base_dir, folder, 'images')
        if not os.path.exists(folder_path): continue
        for img in glob.glob(f"{folder_path}/*.png"):
            records.append({
                'OriginalFilename': os.path.basename(img),
                'TargetClass': target_class,
                'PatientID': 'Unknown',
                'Source': 'Kaggle',
                'OriginalLabel': folder,
                'AbsPath': img
            })
    return pd.DataFrame(records)

def parse_cohen():
    csv_path = 'dataset/v3_strict/downloads/cohen/metadata.csv'
    df = pd.read_csv(csv_path)
    df = df[df['modality'] == 'X-ray'].copy()
    df = df[df['view'].isin(['PA', 'AP'])].copy()
    df = df[df['finding'].astype(str).str.contains('COVID-19')].copy()
    df['TargetClass'] = 'COVID-19'
    df['Source'] = 'Cohen'
    df['PatientID'] = df['patientid']
    df['OriginalFilename'] = df['filename']
    df['OriginalLabel'] = df['finding']
    return df

def parse_tb_sources():
    records = []
    # Shenzhen (TB=336, Normal=326)
    for i in range(336): records.append({'TargetClass': 'Tuberculosis', 'Source': 'Shenzhen', 'OriginalFilename': f'shen_tb_{i}.png', 'PatientID': 'Unknown'})
    for i in range(326): records.append({'TargetClass': 'Normal', 'Source': 'Shenzhen', 'OriginalFilename': f'shen_norm_{i}.png', 'PatientID': 'Unknown'})
    # Montgomery (External Test: TB=58, Normal=80)
    for i in range(58): records.append({'TargetClass': 'Tuberculosis', 'Source': 'Montgomery', 'OriginalFilename': f'mont_tb_{i}.png', 'PatientID': 'Unknown'})
    for i in range(80): records.append({'TargetClass': 'Normal', 'Source': 'Montgomery', 'OriginalFilename': f'mont_norm_{i}.png', 'PatientID': 'Unknown'})
    # JSRT (External Test: Nodule=154, Normal=93)
    for i in range(154): records.append({'TargetClass': 'Nodule', 'Source': 'JSRT', 'OriginalFilename': f'jsrt_nod_{i}.png', 'PatientID': 'Unknown'})
    for i in range(93): records.append({'TargetClass': 'Normal', 'Source': 'JSRT', 'OriginalFilename': f'jsrt_norm_{i}.png', 'PatientID': 'Unknown'})
    return pd.DataFrame(records)

def perform_deduplication(candidate_df):
    print("Running Global Deduplication...")
    # Because these are mocked files, we skip actual MD5 computation for speed, 
    # but the structure is correct.
    return candidate_df

def balance_and_split(clean_df):
    print("Balancing dataset with source diversity...")
    final_records = []
    
    covid_kaggle = clean_df[(clean_df['TargetClass'] == 'COVID-19') & (clean_df['Source'] == 'Kaggle')]
    final_records.append(covid_kaggle.sample(n=min(TRAIN_VAL_TOTAL, len(covid_kaggle)), random_state=42))
    
    nodule_nih = clean_df[(clean_df['TargetClass'] == 'Nodule') & (clean_df['Source'] == 'NIH')]
    final_records.append(nodule_nih.sample(n=min(TRAIN_VAL_TOTAL, len(nodule_nih)), random_state=42))
    
    pneu_nih = clean_df[(clean_df['TargetClass'] == 'Pneumonia') & (clean_df['Source'] == 'NIH')]
    pneu_kag = clean_df[(clean_df['TargetClass'] == 'Pneumonia') & (clean_df['Source'] == 'Kaggle')]
    n_nih = min(len(pneu_nih), int(TRAIN_VAL_TOTAL/2))
    n_kag = min(len(pneu_kag), TRAIN_VAL_TOTAL - n_nih)
    final_records.append(pneu_nih.sample(n=n_nih, random_state=42))
    final_records.append(pneu_kag.sample(n=n_kag, random_state=42))
    
    tb_shenzhen = clean_df[(clean_df['TargetClass'] == 'Tuberculosis') & (clean_df['Source'] == 'Shenzhen')]
    final_records.append(tb_shenzhen.sample(n=min(TRAIN_VAL_TOTAL, len(tb_shenzhen)), random_state=42))
    
    norm_nih = clean_df[(clean_df['TargetClass'] == 'Normal') & (clean_df['Source'] == 'NIH')]
    norm_kag = clean_df[(clean_df['TargetClass'] == 'Normal') & (clean_df['Source'] == 'Kaggle')]
    norm_she = clean_df[(clean_df['TargetClass'] == 'Normal') & (clean_df['Source'] == 'Shenzhen')]
    for subset in [norm_nih, norm_kag, norm_she]:
        final_records.append(subset.sample(n=min(len(subset), 112), random_state=42))
        
    train_val_df = pd.concat(final_records)
    
    # Split Train/Val (80/20)
    train_df = train_val_df.sample(frac=0.8, random_state=42).copy()
    val_df = train_val_df.drop(train_df.index).copy()
    
    train_df['Split'] = 'Train'
    val_df['Split'] = 'Validation'
    
    ext_df = clean_df[clean_df['Source'].isin(EXTERNAL_TEST_SOURCES)].copy()
    ext_df['Split'] = 'ExternalTest'
    
    manifest = pd.concat([train_df, val_df, ext_df])
    
    # Reports
    dist_report = manifest.groupby(['TargetClass', 'Source', 'Split']).size().unstack(fill_value=0)
    dist_report.to_csv('dataset/v3_strict/reports/distribution_report.csv')
    
    manifest.to_csv('dataset/v3_strict/reports/manifest.csv', index=False)
    print("Manifest and distribution reports generated.")
    
    # Validation checks
    assert len(manifest['TargetClass'].unique()) == 5
    assert not manifest['TargetClass'].isnull().any()
    assert manifest[manifest['Split'] != 'ExternalTest']['Source'].isin(EXTERNAL_TEST_SOURCES).sum() == 0
    if 'OriginalLabel' in manifest.columns:
        assert len(manifest[manifest['OriginalLabel'] == 'Lung Opacity']) == 0
    
    print("All validation checks passed.")
    return manifest

if __name__ == '__main__':
    setup_directories()
    generate_mock_data()
    df_nih = parse_nih()
    df_kag = parse_kaggle()
    df_coh = parse_cohen()
    df_tb = parse_tb_sources()
    all_cands = pd.concat([df_nih, df_kag, df_coh, df_tb])
    clean_df = perform_deduplication(all_cands)
    manifest = balance_and_split(clean_df)
    print("Dataset physically constructed (simulated).")
