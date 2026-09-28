import json
import collections

with open('dataset/gdrive_contents.json', encoding='utf-16') as f:
    data = json.load(f)

counts = collections.Counter()
for p in data:
    path = p['path']
    if any(path.endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
        folder = path.split('/')[1] if len(path.split('/')) > 1 else 'root'
        counts[folder] += 1

print("Dataset Audit:")
for k, v in counts.items():
    print(f"{k}: {v}")
