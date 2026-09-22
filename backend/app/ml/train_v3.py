"""
ChestVision V3 Strict Training Script
"""
import argparse
import json
import os
import csv
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, confusion_matrix
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from app.core.config import settings
from app.ml.model import ChestVisionModel

def get_v3_transform(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(contrast=0.15, brightness=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.NORMALIZE_MEAN, std=settings.NORMALIZE_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=settings.NORMALIZE_MEAN, std=settings.NORMALIZE_STD),
        ])

class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_map: dict, train: bool):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.transform = get_v3_transform(train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row['filepath'])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        lbl = row["label"]
        target = self.label_map[lbl]
        return image, torch.tensor(target, dtype=torch.long)


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
    parser.add_argument("--csv", required=True, help="Path to manifest CSV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset manifest
    print(f"Loading manifest from: {args.csv}")
    df = pd.read_csv(args.csv)
    
    label_cols = sorted(df["label"].unique().tolist())
    label_map = {lbl: idx for idx, lbl in enumerate(label_cols)}
    
    print(f"Loaded {len(df)} samples, {len(label_cols)} classes: {label_cols}")

    # Validate all paths
    print("Validating all manifest paths before training...")
    missing = []
    for idx, row in df.iterrows():
        path = str(row['filepath'])
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print(f"CRITICAL ERROR: {len(missing)} files missing. Aborting training.")
        print(f"First 5 missing: {missing[:5]}")
        return
    print("All file paths validated successfully.")

    train_df = df[df['split'] == 'Train'].copy()
    val_df = df[df['split'] == 'Val'].copy()
    test_df = df[df['split'] == 'Test'].copy()

    print(f"Splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Dynamic Class Weights Calculation from the Train set ONLY (Manual inverse frequency)
    train_labels = train_df["label"].values
    counts = Counter(train_labels)
    total = sum(counts.values())
    n_classes = len(label_cols)
    class_weights = []
    for lbl in label_cols:
        weight = total / (n_classes * counts[lbl])
        class_weights.append(weight)
        
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    class_weights_dict = dict(zip(label_cols, class_weights))
    print(f"Calculated Dynamic Class Weights: {class_weights_dict}")

    train_ds = ChestXrayDataset(train_df, label_map, train=True)
    val_ds = ChestXrayDataset(val_df, label_map, train=False)
    test_ds = ChestXrayDataset(test_df, label_map, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ChestVisionModel(num_classes=len(label_cols), pretrained=True)
    
    early_params = []
    later_params = []
    classifier_params = []
    
    for name, param in model.backbone.named_parameters():
        param.requires_grad = True # UNFREEZE ENTIRE BACKBONE
        if "denseblock4" in name or "norm5" in name:
            later_params.append(param)
        elif "classifier" in name:
            classifier_params.append(param)
        else:
            early_params.append(param)
            
    print("All parameters unfrozen for full fine-tuning.")
    
    model = model.to(device)
    
    # Class-weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Discriminative LRs
    optimizer = torch.optim.AdamW([
        {'params': early_params, 'lr': 1e-5},
        {'params': later_params, 'lr': 3e-5},
        {'params': classifier_params, 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    best_val_f1 = -1
    patience_counter = 0
    max_patience = 8
    
    NEW_MODEL_DIR = r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\models_store\new_model_v3"
    os.makedirs(NEW_MODEL_DIR, exist_ok=True)
    
    # Logs directory
    LOGS_DIR = os.path.join(NEW_MODEL_DIR, "training_logs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    BEST_MODEL_PATH = os.path.join(NEW_MODEL_DIR, "best_model.pt")
    FINAL_MODEL_PATH = os.path.join(NEW_MODEL_DIR, "final_model.pt")
    LABELS_PATH = os.path.join(NEW_MODEL_DIR, "labels.json")
    WEIGHTS_PATH = os.path.join(NEW_MODEL_DIR, "class_weights.json")
    METRICS_PATH = os.path.join(NEW_MODEL_DIR, "metrics.json")
    HISTORY_PATH = os.path.join(NEW_MODEL_DIR, "training_history.csv")
    CONFIG_PATH = os.path.join(NEW_MODEL_DIR, "model_config.json")
    
    with open(LABELS_PATH, "w") as f:
        json.dump(label_cols, f)
        
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(class_weights_dict, f, indent=2)
        
    model_config = {
        "architecture": "DenseNet121",
        "num_classes": len(label_cols),
        "pretrained_imagenet": True,
        "input_size": settings.IMAGE_SIZE,
        "optimizer": "AdamW",
        "learning_rates": {"early": 1e-5, "later": 3e-5, "classifier": 1e-4},
        "loss": "CrossEntropyLoss",
        "max_epochs": args.epochs,
        "early_stopping_patience": max_patience
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(model_config, f, indent=2)

    history = []
    
    # Full Convergence Training Loop
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
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss={train_loss:.4f} "
              f"val_accuracy={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f} "
              f"lr={current_lr}")

        scheduler.step(val_metrics['macro_f1'])

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Saved new best model (val_f1={best_val_f1:.4f})")
            early_stopping = False
        else:
            patience_counter += 1
            print(f"  -> No improvement. Early stopping patience: {patience_counter}/{max_patience}")
            early_stopping = patience_counter >= max_patience
            
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_accuracy": val_metrics['accuracy'],
            "val_macro_f1": val_metrics['macro_f1'],
            "lr": current_lr,
            "early_stopping_status": "Triggered" if early_stopping else f"Patience {patience_counter}/{max_patience}"
        })
        
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)
        
        if early_stopping:
            print("Early stopping triggered. Convergence reached.")
            break

    # Save final epoch model
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    
    # Evaluate best checkpoint on Internal Test Set
    print("\nLoading Best Model Checkpoint for Internal Testing...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    test_metrics = evaluate(model, test_loader, device, label_cols)
    test_metrics["label_cols"] = label_cols
    
    test_metrics["model_path"] = BEST_MODEL_PATH
    test_metrics["epochs_completed"] = epoch + 1
    test_metrics["dataset_splits"] = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    test_metrics["file_size_mb"] = os.path.getsize(BEST_MODEL_PATH) / (1024 * 1024)
    
    with open(METRICS_PATH, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("\n=== Final Internal Test Metrics ===")
    print(f"Internal Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Internal Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"\nModel saved to: {NEW_MODEL_DIR}")

if __name__ == "__main__":
    main()
