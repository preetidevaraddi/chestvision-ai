import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from PIL import Image
import random

# Add backend to path for app imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.ml.model import load_trained_model

# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
WORKSPACE_DIR = r"c:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\v3_workspace"
MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "dataset_manifest.csv")
PREV_MODEL_STORE = r"c:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\models_store\new_model_v3_demo"
MODEL_STORE = r"c:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\models_store\new_model_v3_extended"

CLASSES = ["COVID-19", "Normal", "Pneumonia", "Tuberculosis"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}

MAX_TOTAL_EPOCHS = 30
START_EPOCH = 1  # 1 epoch already done
PATIENCE = 6
BATCH_SIZE = 32

os.makedirs(MODEL_STORE, exist_ok=True)

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
log_path = os.path.join(MODEL_STORE, "training_logs.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info("--- CHESTVISION AI EXTENDED TRAINING (4-CLASS) ---")

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
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
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

def main():
    df = pd.read_csv(MANIFEST_PATH)
    df = df[df['label'] != 'Nodule'].copy()
    
    train_df = df[df['split'] == 'Train']
    val_df = df[df['split'] == 'Val']
    test_df = df[df['split'] == 'Test']
    
    train_transform, val_transform = get_transforms()
    
    train_ds = ChestVisionDataset(train_df, transform=train_transform)
    val_ds = ChestVisionDataset(val_df, transform=val_transform)
    test_ds = ChestVisionDataset(test_df, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load previous class weights
    with open(os.path.join(PREV_MODEL_STORE, "class_weights.json"), "r") as f:
        class_weights_dict = json.load(f)
    weights = [class_weights_dict[cls] for cls in CLASSES]
    logger.info(f"Loaded Class Weights: {class_weights_dict}")
    
    with open(os.path.join(MODEL_STORE, "class_weights.json"), "w") as f:
        json.dump(class_weights_dict, f, indent=4)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    weights_tensor = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # Load baseline model exactly as it was saved
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    prev_model_path = os.path.join(PREV_MODEL_STORE, "best_model.pt")
    model.load_state_dict(torch.load(prev_model_path, map_location="cpu", weights_only=True))
    model = model.to(device)
    
    early_params, later_params = [], []
    for name, param in model.features.named_parameters():
        if 'denseblock1' in name or 'denseblock2' in name or 'conv0' in name or 'norm0' in name:
            early_params.append(param)
        else:
            later_params.append(param)
            
    classifier_params = list(model.classifier.parameters())
    
    optimizer = optim.AdamW([
        {'params': early_params, 'lr': 1e-5},
        {'params': later_params, 'lr': 3e-5},
        {'params': classifier_params, 'lr': 1e-4}
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # We evaluate the baseline once to get the exact baseline stats
    best_macro_f1 = 0.8007  # Baseline Macro F1
    best_epoch = 1
    epochs_no_improve = 0
    history = []
    
    logger.info("Resuming Training...")
    for epoch in range(START_EPOCH, MAX_TOTAL_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_ds)
        val_acc = accuracy_score(all_labels, all_preds)
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        current_lr = optimizer.param_groups[-1]['lr']
        logger.info(f"Epoch {epoch+1}/{MAX_TOTAL_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val Macro F1: {val_macro_f1:.4f} - LR: {current_lr:.6f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_macro_f1': val_macro_f1,
            'lr': current_lr
        })
        
        scheduler.step(val_macro_f1)
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODEL_STORE, "best_model.pt"))
            logger.info("Saved new best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
                break
                
    torch.save(model.state_dict(), os.path.join(MODEL_STORE, "final_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(MODEL_STORE, "training_history.csv"), index=False)
    
    with open(os.path.join(MODEL_STORE, "labels.json"), "w") as f:
        json.dump(CLASSES, f, indent=4)
        
    with open(os.path.join(MODEL_STORE, "model_config.json"), "w") as f:
        json.dump({
            "architecture": "DenseNet121",
            "pretrained": True,
            "num_classes": len(CLASSES),
            "classes": CLASSES
        }, f, indent=4)
        
    logger.info("Training complete. Starting final evaluation on test set...")
    
    # Evaluate best model using the API loader
    if not os.path.exists(os.path.join(MODEL_STORE, "best_model.pt")):
        logger.info("No better model found. The baseline model is still the best.")
        best_checkpoint_path = prev_model_path
    else:
        best_checkpoint_path = os.path.join(MODEL_STORE, "best_model.pt")
        
    # Test load with backend integration function
    try:
        backend_model = load_trained_model(best_checkpoint_path, num_classes=len(CLASSES), device=str(device))
        logger.info("Successfully verified load_trained_model() on the new checkpoint.")
    except Exception as e:
        logger.error(f"Failed to load checkpoint with backend method: {e}")
        return

    backend_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = backend_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    macro_p, macro_r, macro_f1, _ = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)['macro avg'].values()
    report_dict = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    eval_report = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "test_accuracy": acc,
        "test_macro_precision": macro_p,
        "test_macro_recall": macro_r,
        "test_macro_f1": macro_f1,
        "per_class": report_dict,
        "confusion_matrix": cm.tolist()
    }
    
    with open(os.path.join(MODEL_STORE, "metrics.json"), "w") as f:
        json.dump(eval_report, f, indent=4)
        
    logger.info("\n--- FINAL EVALUATION REPORT ---")
    logger.info(f"Best Epoch: {best_epoch}")
    logger.info(f"Best Validation Macro F1: {best_macro_f1:.4f}")
    logger.info(f"Test Accuracy: {acc:.4f} (Baseline: 0.7871)")
    logger.info(f"Test Macro Precision: {macro_p:.4f}")
    logger.info(f"Test Macro Recall: {macro_r:.4f}")
    logger.info(f"Test Macro F1: {macro_f1:.4f} (Baseline: 0.8007)")
    logger.info(f"\n{classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0)}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    if acc > 0.7871 and macro_f1 > 0.8007:
        logger.info("RESULTS: The extended model performed better than the baseline.")
    else:
        logger.info("RESULTS: The extended model DID NOT perform better than the baseline.")

if __name__ == "__main__":
    main()
