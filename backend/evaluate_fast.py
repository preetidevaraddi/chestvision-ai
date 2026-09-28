import os
import json
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from app.ml.model import load_trained_model
from app.ml.preprocessing import get_transform
from PIL import Image

class EvalDataset(Dataset):
    def __init__(self, df, label_map):
        self.df = df
        self.label_map = label_map
        self.transform = get_transform(train=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join('dataset/images', row['filename'])
        if not os.path.exists(img_path):
            img_path = os.path.join('dataset/real_images', row['filename'])
            
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, self.label_map[row["label"]], row["filename"], row.get("source", "unknown")

def evaluate_model_fast(model_dir, out_file):
    print(f"Evaluating model in {model_dir}")
    df = pd.read_csv('dataset/labels_cleaned.csv')
    
    # Imitate split logic to find unused images
    min_count = df['label'].value_counts().min()
    balanced_dfs = [df[df['label'] == cls].head(min_count) for cls in df['label'].unique()]
    bdf = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    used_filenames = bdf['filename'].tolist()
    unused_df = df[~df['filename'].isin(used_filenames)].copy()
    
    # Cap at 50 per class for speed
    eval_df = pd.DataFrame()
    for cls in unused_df['label'].unique():
        cls_df = unused_df[unused_df['label'] == cls]
        eval_df = pd.concat([eval_df, cls_df.head(50)]) 
        
    labels_path = os.path.join(model_dir, "labels.json")
    model_path = os.path.join(model_dir, "chestvision_model.pt")
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
        
    with open(labels_path, "r") as f:
        labels = json.load(f)
        
    label_map = {lbl: i for i, lbl in enumerate(labels)}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_trained_model(model_path, len(labels), device)
    model.eval()
    
    ds = EvalDataset(eval_df, label_map)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    y_true = []
    y_pred = []
    failures = []
    
    with torch.no_grad():
        for images, targets, filenames, sources in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            confs = torch.max(probs, dim=1)[0].cpu().numpy()
            targets = targets.numpy()
            
            y_true.extend(targets)
            y_pred.extend(preds)
            
            for i in range(len(preds)):
                if preds[i] != targets[i]:
                    failures.append({
                        'filename': filenames[i],
                        'true_class': labels[targets[i]],
                        'predicted_class': labels[preds[i]],
                        'confidence': float(confs[i]),
                        'source': sources[i]
                    })
                    
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)), average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    results = {
        'accuracy': float(acc),
        'macro_precision': float(prec_macro),
        'macro_recall': float(rec_macro),
        'macro_f1': float(f1_macro),
        'per_class': {
            labels[i]: {
                'precision': float(prec_per[i]),
                'recall': float(rec_per[i]),
                'f1': float(f1_per[i])
            } for i in range(len(labels))
        },
        'confusion_matrix': cm.tolist(),
        'cm_labels': labels,
        'failures': failures
    }
    
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Evaluation completed for {model_dir}. Results saved to {out_file}.")

if __name__ == '__main__':
    evaluate_model_fast('models_store/new_model_v3', 'eval_results_v3.json')
