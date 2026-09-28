import json
import os
import random
import requests

def download_file(url, out_path):
    # Google Drive direct download URL format for small files
    file_id = url.split('id=')[-1]
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        r = requests.get(direct_url, timeout=10)
        if r.status_code == 200 and 'html' not in r.headers.get('Content-Type', ''):
            with open(out_path, 'wb') as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

def main():
    os.makedirs('dataset/real_images', exist_ok=True)
    with open('dataset/gdrive_contents.json', encoding='utf-16') as f:
        data = json.load(f)

    tb_links = [p for p in data if 'tuberculosis' in p['path'].lower() and p['path'].endswith('.png')]
    pneu_links = [p for p in data if 'pneumonia' in p['path'].lower() and p['path'].endswith('.png')]

    random.shuffle(tb_links)
    random.shuffle(pneu_links)

    print("Fetching TB...")
    count = 0
    for p in tb_links:
        if count >= 15: break
        if download_file(p['url'], f"dataset/real_images/Tuberculosis_ext_{count:04d}.png"):
            count += 1
            print(f"TB: {count}/15")

    print("Fetching Pneumonia...")
    count = 0
    for p in pneu_links:
        if count >= 15: break
        if download_file(p['url'], f"dataset/real_images/Pneumonia_ext_{count:04d}.png"):
            count += 1
            print(f"Pneumonia: {count}/15")

if __name__ == '__main__':
    main()
