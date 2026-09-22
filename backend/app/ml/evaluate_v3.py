"""
V3 Standalone Checkpoint Validator and Evaluator
"""
import json
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from PIL import Image
import torchvision.transforms as transforms

from app.ml.model import ChestVisionModel
from app.core.config import settings
from app.ml.train_v3 import ChestXrayDataset

def get_v3_eval_transform():
    return transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.NORMALIZE_MEAN, std=settings.NORMALIZE_STD),
    ])

def evaluate_loader(model, loader, device, label_cols):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    targets = np.array(all_targets)
    preds = np.array(all_preds)

    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0
    )
    
    cm = confusion_matrix(targets, preds, labels=range(len(label_cols)))
    
    report = classification_report(targets, preds, target_names=label_cols, zero_division=0, output_dict=True)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "report": report
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading V3 for Evaluation on {device}...")
    
    V3_DIR = r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\models_store\new_model_v3"
    MANIFEST_PATH = r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\v3_workspace\dataset_manifest.csv"
    
    with open(os.path.join(V3_DIR, "labels.json"), "r") as f:
        label_cols = json.load(f)
    label_map = {lbl: idx for idx, lbl in enumerate(label_cols)}

    # Verify checkpoint loading
    model = ChestVisionModel(num_classes=len(label_cols), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(V3_DIR, "best_model.pt"), map_location=device))
    model = model.to(device)
    print("✓ Checkpoint loaded successfully in a fresh process.")
    
    df = pd.read_csv(MANIFEST_PATH)
    val_df = df[df['split'] == 'Val']
    test_df = df[df['split'] == 'Test']
    ext_df = df[df['is_external_test'] == True]
    
    val_ds = ChestXrayDataset(val_df, label_map, train=False)
    test_ds = ChestXrayDataset(test_df, label_map, train=False)
    ext_ds = ChestXrayDataset(ext_df, label_map, train=False)
    
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    ext_loader = DataLoader(ext_ds, batch_size=16, shuffle=False)
    
    print("Evaluating Internal Val Set...")
    val_res = evaluate_loader(model, val_loader, device, label_cols)
    print("Evaluating Internal Test Set...")
    test_res = evaluate_loader(model, test_loader, device, label_cols)
    print("Evaluating External Test Cohort...")
    ext_res = evaluate_loader(model, ext_loader, device, label_cols)
    
    # UNSEEN BENCHMARK (V2 Unseen List)
    print("Evaluating Unseen Benchmark...")
    old_labels = pd.read_csv(r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\labels.csv")
    class_counts = old_labels['label'].value_counts()
    min_count = class_counts.min()
    balanced_dfs = [old_labels[old_labels['label'] == cls].head(min_count) for cls in class_counts.index]
    used_df = pd.concat(balanced_dfs)
    unused_df = old_labels[~old_labels['filename'].isin(used_df['filename'])].copy()
    
    # Try to map paths for unused_df
    valid_unseen = []
    for _, row in unused_df.iterrows():
        img_path = os.path.join(r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\images", row['filename'])
        if not os.path.exists(img_path):
            img_path = os.path.join(r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\real_images", row['filename'])
        if os.path.exists(img_path):
            valid_unseen.append({'filepath': img_path, 'label': row['label']})
    
    unseen_res = None
    if valid_unseen:
        unseen_df = pd.DataFrame(valid_unseen)
        # Only evaluate classes that exist in V3
        unseen_df = unseen_df[unseen_df['label'].isin(label_cols)]
        if not unseen_df.empty:
            unseen_ds = ChestXrayDataset(unseen_df, label_map, train=False)
            unseen_loader = DataLoader(unseen_ds, batch_size=16, shuffle=False)
            unseen_res = evaluate_loader(model, unseen_loader, device, label_cols)
            
    # Compile final evaluation report
    eval_report = {
        "Val": val_res,
        "Test": test_res,
        "ExternalTest": ext_res,
        "UnseenBenchmark": unseen_res
    }
    
    with open(os.path.join(V3_DIR, "v3_evaluation_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)
        
    print("\n=== V3 EVALUATION SUMMARY ===")
    print(f"Internal Val Macro F1: {val_res['macro_f1']:.4f}")
    print(f"Internal Test Macro F1: {test_res['macro_f1']:.4f} (Accuracy: {test_res['accuracy']:.4f})")
    print(f"External Test Macro F1: {ext_res['macro_f1']:.4f}")
    if unseen_res:
        print(f"Unseen Benchmark Macro F1: {unseen_res['macro_f1']:.4f} (Accuracy: {unseen_res['accuracy']:.4f})")
    print("V2 Comparison targets: Test Acc 82.55%, Test F1 82.63%, Unseen Acc ~86.5%")

if __name__ == "__main__":
    main()
