import json
import os
import subprocess

def main():
    os.makedirs('dataset/zips', exist_ok=True)
    os.makedirs('dataset/raw', exist_ok=True)
    with open('dataset/gdrive_contents.json', encoding='utf-16') as f:
        data = json.load(f)

    zips_to_download = ['COVID19.zip', 'Normal.zip', 'Pneumonia.zip', 'Tuberculosis.zip', 'LungCancer.zip']
    
    for z in zips_to_download:
        # Find url
        url = None
        for p in data:
            if p['path'].endswith(z):
                url = p['url']
                break
        
        if url:
            out_path = f"dataset/zips/{z}"
            if not os.path.exists(out_path):
                print(f"Downloading {z}...")
                subprocess.run(['python', '-m', 'gdown', url, '-O', out_path, '--quiet'])
            else:
                print(f"{z} already exists.")
        else:
            print(f"URL for {z} not found.")

if __name__ == '__main__':
    main()
