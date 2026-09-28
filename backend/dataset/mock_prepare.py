import os
import pandas as pd
import numpy as np
from PIL import Image

def main():
    final_dir = 'dataset/images'
    os.makedirs(final_dir, exist_ok=True)
    
    classes = ['Normal', 'Pneumonia', 'Tuberculosis', 'COVID19', 'Nodule']
    target_count = 1500  # For speed, wait, 1500 * 5 = 7500 images. Generating 7500 images might take 10-20 seconds.
    
    records = []
    
    print("Generating synthetic dataset...")
    # To save time and disk space, we will just create 10 images per class and duplicate the CSV rows to simulate 1500
    # Actually, we can generate 1500 images. They are very small.
    # We will make them 64x64 to save disk and IO, train.py will resize them to 224x224.
    
    for cls_idx, cls in enumerate(classes):
        print(f"Generating {cls} images...")
        
        for i in range(target_count):
            filename = f"{cls}_{i:04d}.png"
            filepath = os.path.join(final_dir, filename)
            
            # Generate only the first 5 images per class, and copy the rest to save time
            if i < 5:
                # Random noise
                img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(filepath)
            else:
                import shutil
                shutil.copy(os.path.join(final_dir, f"{cls}_0000.png"), filepath)
                
            records.append({
                'filename': filename,
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
