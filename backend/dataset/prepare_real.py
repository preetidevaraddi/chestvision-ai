import os
import shutil
import pandas as pd
from PIL import Image

def main():
    final_dir = 'dataset/images'
    os.makedirs(final_dir, exist_ok=True)
    
    # 1. Process COVID from covid_github
    covid_images_dir = 'dataset/covid_github/images'
    covid_metadata = 'dataset/covid_github/metadata.csv'
    
    records = []
    classes_counts = {'Normal': 0, 'Pneumonia': 0, 'Tuberculosis': 0, 'COVID19': 0, 'Nodule': 0}
    
    if os.path.exists(covid_metadata):
        df_covid = pd.read_csv(covid_metadata)
        # Filter for COVID-19 and PA/AP view
        df_covid = df_covid[df_covid['finding'].str.contains('COVID-19', na=False)]
        
        # We only take up to 100 images
        df_covid = df_covid.head(100)
        for _, row in df_covid.iterrows():
            filename = row['filename']
            src = os.path.join(covid_images_dir, filename)
            if os.path.exists(src):
                dst_name = f"COVID19_{classes_counts['COVID19']:04d}.png"
                dst = os.path.join(final_dir, dst_name)
                # Convert to PNG for consistency and ignore bad images
                try:
                    img = Image.open(src).convert('RGB')
                    img.save(dst)
                    classes_counts['COVID19'] += 1
                    records.append({
                        'filename': dst_name,
                        'Normal': 0, 'Pneumonia': 0, 'Tuberculosis': 0, 'COVID-19': 1, 'Nodule': 0
                    })
                except Exception as e:
                    print(f"Skipping {src}: {e}")
                    
    # 2. Process other classes from real_images
    real_images_dir = 'dataset/real_images'
    if os.path.exists(real_images_dir):
        for f in os.listdir(real_images_dir):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                cls = None
                if f.startswith('Normal'): cls = 'Normal'
                elif f.startswith('Pneumonia'): cls = 'Pneumonia'
                elif f.startswith('Tuberculosis'): cls = 'Tuberculosis'
                elif f.startswith('Nodule'): cls = 'Nodule'
                
                if cls:
                    src = os.path.join(real_images_dir, f)
                    dst_name = f"{cls}_{classes_counts[cls]:04d}.png"
                    dst = os.path.join(final_dir, dst_name)
                    try:
                        img = Image.open(src).convert('RGB')
                        img.save(dst)
                        classes_counts[cls] += 1
                        records.append({
                            'filename': dst_name,
                            'Normal': 1 if cls == 'Normal' else 0,
                            'Pneumonia': 1 if cls == 'Pneumonia' else 0,
                            'Tuberculosis': 1 if cls == 'Tuberculosis' else 0,
                            'COVID-19': 0,
                            'Nodule': 1 if cls == 'Nodule' else 0
                        })
                    except Exception as e:
                        print(f"Skipping {src}: {e}")
                        
    # Balance classes by capping at the minimum count
    min_count = min(classes_counts.values())
    print(f"Balancing all classes to {min_count} images.")
    
    df = pd.DataFrame(records)
    
    balanced_dfs = []
    for cls in ['Normal', 'Pneumonia', 'Tuberculosis', 'COVID-19', 'Nodule']:
        cls_df = df[df[cls] == 1].head(min_count)
        balanced_dfs.append(cls_df)
        
    df = pd.concat(balanced_dfs).drop_duplicates('filename').reset_index(drop=True)
        
    df.to_csv('dataset/labels.csv', index=False)
    print(f"Created dataset/labels.csv with {len(df)} rows.")
    print("Class distribution:", classes_counts)

if __name__ == '__main__':
    main()
