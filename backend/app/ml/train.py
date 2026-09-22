"""
Standalone model training script for single-label, mutually exclusive 5-class problem.

Run with:
    python -m app.ml.train --csv dataset/labels.csv --image-dir dataset/images

Expected labels.csv format:
    filename,label
    (where label is the string class name, e.g. "Normal")
"""
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from app.core.config import settings
from app.ml.preprocessing import get_transform
from app.ml.model import ChestVisionModel
from PIL import Image


class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, label_map: dict, train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.label_map = label_map
        self.transform = get_transform(train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # CrossEntropyLoss expects a long integer target
        target = self.label_map[row["label"]]
        return image, torch.tensor(target, dtype=torch.long)


def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, List[str], dict]:
    df = pd.read_csv(csv_path)
    label_cols = sorted(df["label"].unique().tolist())
    label_map = {lbl: idx for idx, lbl in enumerate(label_cols)}
    return df, label_cols, label_map


def evaluate(model, loader, device, label_cols) -> dict:
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
    
    # Generate confusion matrix
    cm = confusion_matrix(targets, preds, labels=range(len(label_cols)))

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class_precision": per_class_p.tolist(),
        "per_class_recall": per_class_r.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix": cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to labels CSV")
    parser.add_argument("--image-dir", required=True, help="Directory containing the images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df, label_cols, label_map = load_dataset(args.csv)
    print(f"Loaded {len(df)} samples, {len(label_cols)} classes: {label_cols}")

    # Find minimum class count
    class_counts = df['label'].value_counts()
    min_count = class_counts.min()
    print(f"Balancing dataset to {min_count} unique real images per class...")
    
    balanced_dfs = []
    for cls in class_counts.index:
        balanced_dfs.append(df[df['label'] == cls].head(min_count))
    df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    # For patient-aware split, check if patientid exists.
    if 'patientid' in df.columns:
        print("Performing patient-aware split...")
        unique_patients = df['patientid'].unique()
        train_p, temp_p = train_test_split(unique_patients, test_size=0.3, random_state=42)
        val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)
        
        train_df = df[df['patientid'].isin(train_p)]
        val_df = df[df['patientid'].isin(val_p)]
        test_df = df[df['patientid'].isin(test_p)]
    else:
        print("No patientid found. Performing random split...")
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_ds = ChestXrayDataset(train_df, args.image_dir, label_map, train=True)
    val_ds = ChestXrayDataset(val_df, args.image_dir, label_map, train=False)
    test_ds = ChestXrayDataset(test_df, args.image_dir, label_map, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ChestVisionModel(num_classes=len(label_cols))
    
    # Freeze the DenseNet feature backbone for CPU training
    for name, param in model.backbone.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    print("Frozen DenseNet121 backbone. Only training classifier.")
    
    model = model.to(device)
    
    # Mutually exclusive single-label loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

    best_val_f1 = -1
    patience_counter = 0
    max_patience = 3
    os.makedirs(settings.MODEL_DIR, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device, label_cols)
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss={train_loss:.4f} "
              f"val_accuracy={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}")

        scheduler.step(val_metrics['macro_f1'])

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), settings.MODEL_PATH)
            with open(settings.LABELS_PATH, "w") as f:
                json.dump(label_cols, f)
            print(f"  -> Saved new best model (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Early stopping patience: {patience_counter}/{max_patience}")
            if patience_counter >= max_patience:
                print("Early stopping triggered.")
                break

    # Final evaluation on held-out test set using the best saved model
    model.load_state_dict(torch.load(settings.MODEL_PATH, map_location=device))
    test_metrics = evaluate(model, test_loader, device, label_cols)
    test_metrics["label_cols"] = label_cols
    
    # Add training configuration and model path to metrics for reporting
    test_metrics["model_path"] = settings.MODEL_PATH
    test_metrics["epochs_completed"] = args.epochs
    test_metrics["dataset_splits"] = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    test_metrics["file_size_mb"] = os.path.getsize(settings.MODEL_PATH) / (1024 * 1024)
    
    with open(settings.METRICS_PATH, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n=== Final Test Metrics ===")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nModel saved to: {settings.MODEL_PATH}")
    print(f"Labels saved to: {settings.LABELS_PATH}")
    print(f"Metrics saved to: {settings.METRICS_PATH}")


if __name__ == "__main__":
    main()
