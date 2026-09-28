import os
import hashlib
import pandas as pd

def get_hash(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

df = pd.read_csv('dataset/labels.csv')
hashes = {}
duplicates = []

for _, row in df.iterrows():
    img_path = os.path.join('dataset/images', row['filename'])
    if not os.path.exists(img_path):
        img_path = os.path.join('dataset/real_images', row['filename'])
        if not os.path.exists(img_path):
            continue
    h = get_hash(img_path)
    if h in hashes:
        duplicates.append((row['filename'], hashes[h]))
    else:
        hashes[h] = row['filename']

print(f"Found {len(duplicates)} exact file duplicates.")
if duplicates:
    print("Examples:")
    for d in duplicates[:5]:
        print(f"{d[0]} is identical to {d[1]}")
