import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def download_file(url, out_path):
    if not os.path.exists(out_path):
        subprocess.run(['python', '-m', 'gdown', url, '-O', out_path, '--quiet'])

def main():
    os.makedirs('dataset/real_images', exist_ok=True)
    with open('dataset/gdrive_contents.json', encoding='utf-16') as f:
        data = json.load(f)

    target_count = 100
    classes = {
        'Normal': 0,
        'Pneumonia': 0,
        'Tuberculosis': 0,
        'Nodule': 0
    }
    
    downloads = []
    
    # 1. Start COVID19.zip download
    covid_zip_url = next(p['url'] for p in data if p['path'].endswith('COVID19.zip'))
    covid_zip_path = 'dataset/COVID19.zip'
    downloads.append((covid_zip_url, covid_zip_path))

    # 2. Collect 100 urls for the others
    for p in data:
        path = p['path']
        if not path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        cls = None
        if 'normal' in path.lower(): cls = 'Normal'
        elif 'pneumonia' in path.lower(): cls = 'Pneumonia'
        elif 'tuberculosis' in path.lower(): cls = 'Tuberculosis'
        elif 'lungcancer' in path.lower(): cls = 'Nodule'
        
        if cls and classes[cls] < target_count:
            ext = os.path.splitext(path)[1]
            out_path = f"dataset/real_images/{cls}_{classes[cls]:04d}{ext}"
            downloads.append((p['url'], out_path))
            classes[cls] += 1

    print(f"Starting {len(downloads)} downloads in parallel...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_file, url, out) for url, out in downloads]
        for _ in futures:
            pass # wait for all

if __name__ == '__main__':
    main()
