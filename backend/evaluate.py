import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import sys

# Add backend to path to import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.ml.model import load_trained_model

MODEL_STORE = r"c:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\models_store\new_model_v3_extended"
MANIFEST_PATH = r"c:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\v3_workspace\dataset_manifest.csv"
MODEL_PATH = os.path.join(MODEL_STORE, "best_model.pt")
LABELS_PATH = os.path.join(MODEL_STORE, "labels.json")

# Load labels
with open(LABELS_PATH, "r") as f:
    CLASSES = json.load(f)

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

class ChestVisionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        label_str = row['label']
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        label_idx = CLASS_TO_IDX[label_str]
        return image, label_idx

def get_transforms():
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return val_test_transform

def main():
    print("Loading test dataset...")
    df = pd.read_csv(MANIFEST_PATH)
    # Filter Nodule and use Test split
    df = df[(df['label'] != 'Nodule') & (df['split'] == 'Test')].copy()
    
    val_transform = get_transforms()
    test_ds = ChestVisionDataset(df, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Test dataset size: {len(test_ds)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = load_trained_model(MODEL_PATH, num_classes=len(CLASSES), device=str(device))
    
    print("Evaluating model...")
    all_preds = []
    all_labels = []
    
    # We don't need gradients for evaluation
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            # Predict
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (i+1) % 10 == 0:
                print(f"Batch {i+1}/{len(test_loader)} processed.")
                
    print("Evaluation complete. Calculating metrics...")
    
    acc = accuracy_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    macro_p = report_dict['macro avg']['precision']
    macro_r = report_dict['macro avg']['recall']
    macro_f1 = report_dict['macro avg']['f1-score']
    cm = confusion_matrix(all_labels, all_preds)
    
    eval_report = {
        "test_accuracy": acc,
        "test_macro_precision": macro_p,
        "test_macro_recall": macro_r,
        "test_macro_f1": macro_f1,
        "per_class": report_dict,
        "confusion_matrix": cm.tolist()
    }
    
    metrics_path = os.path.join(MODEL_STORE, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(eval_report, f, indent=4)
        
    print("\n--- RESULTS ---")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall: {macro_r:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\nPer Class Performance:")
    for cls in CLASSES:
        cls_p = report_dict[cls]['precision']
        cls_r = report_dict[cls]['recall']
        cls_f1 = report_dict[cls]['f1-score']
        print(f"  {cls} -> Precision: {cls_p:.4f} | Recall: {cls_r:.4f} | F1: {cls_f1:.4f}")
        
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nExact checkpoint used: {MODEL_PATH}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == '__main__':
    main()
