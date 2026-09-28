import json
import os
import random
import requests
import time
import shutil
import pandas as pd

def download_file(url, out_path, max_retries=3):
    if os.path.exists(out_path):
        return True
    
    file_id = url.split('id=')[-1]
    direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    for attempt in range(max_retries):
        try:
            r = requests.get(direct_url, timeout=15)
            # If rate limited or quota exceeded, Google often returns an HTML page
            if r.status_code == 200 and 'html' not in r.headers.get('Content-Type', '').lower():
                with open(out_path, 'wb') as f:
                    f.write(r.content)
                return True
            else:
                time.sleep(2) # Backoff
        except Exception as e:
            time.sleep(2)
            pass
    return False

def main():
    target_count = 247
    os.makedirs('dataset/images', exist_ok=True)
    
    with open('dataset/gdrive_contents.json', encoding='utf-16') as f:
        data = json.load(f)

    # Filter links
    tb_links = [p for p in data if 'tuberculosis' in p['path'].lower() and p['path'].endswith('.png')]
    pneu_links = [p for p in data if 'pneumonia' in p['path'].lower() and p['path'].endswith('.png')]
    norm_links = [p for p in data if 'normal' in p['path'].lower() and p['path'].endswith('.png')]
    nodule_links = [p for p in data if ('jsrt' in p['path'].lower() or 'cancer' in p['path'].lower()) and p['path'].endswith('.png')]
    
    # Shuffle so we get a random subset
    random.seed(42)
    random.shuffle(tb_links)
    random.shuffle(pneu_links)
    random.shuffle(norm_links)
    random.shuffle(nodule_links)
    
    classes_to_fetch = {
        'Tuberculosis': tb_links,
        'Pneumonia': pneu_links,
        'Normal': norm_links,
        'Nodule': nodule_links
    }
    
    records = []
    
    for cls_name, links in classes_to_fetch.items():
        print(f"Fetching {cls_name} (Target: {target_count})...")
        count = 0
        
        # Check already downloaded to resume
        for file in os.listdir('dataset/images'):
            if file.startswith(cls_name):
                count += 1
                records.append({'filename': file, 'label': cls_name})
                
        for p in links:
            if count >= target_count:
                break
            filename = f"{cls_name}_{count:04d}.png"
            out_path = f"dataset/images/{filename}"
            if download_file(p['url'], out_path):
                count += 1
                records.append({'filename': filename, 'label': cls_name})
                if count % 10 == 0:
                    print(f"  {count}/{target_count} downloaded for {cls_name}")
        print(f"Total {cls_name} obtained: {count}")

    # For COVID-19, we pull from dataset/covid_github
    print("Fetching COVID-19 from local github repo (Target: 247)...")
    covid_df = pd.read_csv('dataset/covid_github/metadata.csv')
    covid_df = covid_df[(covid_df['finding'].str.contains('COVID-19', na=False)) & (covid_df['view'] == 'PA')]
    
    count = 0
    # Add already downloaded
    for file in os.listdir('dataset/images'):
        if file.startswith("COVID-19"):
            count += 1
            # We don't have patient IDs for already downloaded ones here, but wait, let's just 
            # re-copy them to get patient IDs correctly in the CSV. 
            # Actually, let's delete existing COVID ones and recopy to be safe about patientid mapping
            os.remove(f"dataset/images/{file}")
            count -= 1

    covid_records = []
    for _, row in covid_df.iterrows():
        if count >= target_count:
            break
        src_path = os.path.join('dataset/covid_github/images', row['filename'])
        if os.path.exists(src_path):
            dst_filename = f"COVID-19_{count:04d}.png"
            dst_path = os.path.join('dataset/images', dst_filename)
            shutil.copy(src_path, dst_path)
            
            pid = row['patientid']
            covid_records.append({'filename': dst_filename, 'label': 'COVID-19', 'patientid': pid})
            count += 1
            
    print(f"Total COVID-19 obtained: {count}")
    
    # Other records don't have patientid, just put 'unknown_N' to keep them separate in patient splits
    for idx, rec in enumerate(records):
        rec['patientid'] = f"unknown_{idx}"
        
    all_records = records + covid_records
    df = pd.DataFrame(all_records)
    
    # Save the labels.csv
    df.to_csv('dataset/labels.csv', index=False)
    print(f"Saved labels.csv with {len(df)} rows.")

if __name__ == '__main__':
    main()
