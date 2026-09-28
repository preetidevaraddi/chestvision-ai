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

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
WORKSPACE_DIR = r"c:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\v3_workspace"
MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "dataset_manifest.csv")
SUMMARY_PATH = os.path.join(WORKSPACE_DIR, "dataset_summary.csv")
MODEL_STORE = r"c:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\models_store\new_model_v3"

CLASSES = ["COVID-19", "Nodule", "Normal", "Pneumonia", "Tuberculosis"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}

MAX_EPOCHS = 50
PATIENCE = 8
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

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class ChestVisionDataset(Dataset):
    def __init__(self, manifest_df, transform=None):
        self.df = manifest_df.reset_index(drop=True)
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
    # Mild realistic augmentation, NO flips per user instructions
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

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
def train_model():
    logger.info("Starting ChestVision V3 Training Pipeline...")
    
    # Load manifest
    df = pd.read_csv(MANIFEST_PATH)
    
    train_df = df[df['split'] == 'Train']
    val_df = df[df['split'] == 'Val']
    test_df = df[df['split'] == 'Test']
    jsrt_test_df = df[(df['split'] == 'Test') & (df['source'] == 'JSRT')]
    
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Val size: {len(val_df)}")
    logger.info(f"Test size: {len(test_df)}")
    logger.info(f"JSRT Ext Test size: {len(jsrt_test_df)}")
    
    train_transform, val_transform = get_transforms()
    
    train_ds = ChestVisionDataset(train_df, transform=train_transform)
    val_ds = ChestVisionDataset(val_df, transform=val_transform)
    test_ds = ChestVisionDataset(test_df, transform=val_transform)
    jsrt_test_ds = ChestVisionDataset(jsrt_test_df, transform=val_transform)
    
    # Set num_workers to 0 for Windows compatibility safely
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    jsrt_test_loader = DataLoader(jsrt_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Dynamic class weights
    logger.info("Calculating dynamic class weights...")
    label_counts = train_df['label'].value_counts()
    n_samples = len(train_df)
    n_classes = len(CLASSES)
    
    weights = []
    for cls in CLASSES:
        count = label_counts.get(cls, 0)
        if count == 0:
            logger.warning(f"Class '{cls}' has ZERO samples in Train split! Setting weight to 0.0 to avoid ZeroDivisionError.")
            weights.append(0.0)
        else:
            w = n_samples / (n_classes * count)
            weights.append(w)
            
    logger.info(f"Class weights: {weights}")
    
    # Save weights
    with open(os.path.join(MODEL_STORE, "class_weights.json"), "w") as f:
        json.dump({cls: w for cls, w in zip(CLASSES, weights)}, f, indent=4)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    weights_tensor = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # Model Setup
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, n_classes)
    model = model.to(device)
    
    # Differential learning rates
    # Early backbone (features.denseblock1 and denseblock2)
    early_params = []
    # Later backbone (features.denseblock3 and denseblock4, norm5)
    later_params = []
    
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
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_macro_f1 = 0.0
    epochs_no_improve = 0
    history = []
    
    logger.info("Starting Training Loop...")
    for epoch in range(MAX_EPOCHS):
        # Train
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
            
        train_loss = train_loss / len(train_ds)
        
        # Validate
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
                
        val_loss = val_loss / len(val_ds)
        
        # Evaluate validation metrics
        # macro F1 using zero_division=0 to handle classes with 0 instances (like Nodule)
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        logger.info(f"Epoch {epoch+1}/{MAX_EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Macro F1: {val_macro_f1:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_macro_f1': val_macro_f1
        })
        
        scheduler.step(val_macro_f1)
        
        # Check early stopping & save best
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODEL_STORE, "best_model.pt"))
            logger.info("Saved new best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                logger.info(f"Early stopping triggered after {PATIENCE} epochs of no improvement.")
                break
                
    # Save final model & history
    torch.save(model.state_dict(), os.path.join(MODEL_STORE, "final_model.pt"))
    pd.DataFrame(history).to_csv(os.path.join(MODEL_STORE, "training_history.csv"), index=False)
    
    # Save configs
    with open(os.path.join(MODEL_STORE, "labels.json"), "w") as f:
        json.dump(IDX_TO_CLASS, f, indent=4)
        
    with open(os.path.join(MODEL_STORE, "model_config.json"), "w") as f:
        json.dump({
            "architecture": "DenseNet121",
            "pretrained": True,
            "num_classes": n_classes,
            "classes": CLASSES
        }, f, indent=4)
        
    # Copy dataset files for provenance
    import shutil
    shutil.copy(MANIFEST_PATH, os.path.join(MODEL_STORE, "dataset_manifest.csv"))
    shutil.copy(SUMMARY_PATH, os.path.join(MODEL_STORE, "dataset_summary.csv"))
    
    return best_macro_f1

def evaluate_model(split_name, model, dataloader, device):
    logger.info(f"Evaluating on {split_name}...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    report_dict = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True, zero_division=0)
    report_str = classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    logger.info(f"--- {split_name} Evaluation ---")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"\n{report_str}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = models.densenet121()
    model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    model.load_state_dict(torch.load(os.path.join(MODEL_STORE, "best_model.pt"), map_location=device, weights_only=True))
    model = model.to(device)
    
    df = pd.read_csv(MANIFEST_PATH)
    _, val_transform = get_transforms()
    
    # 1. Internal Test
    test_df = df[df['split'] == 'Test']
    test_ds = ChestVisionDataset(test_df, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_metrics = evaluate_model("Internal_Test", model, test_loader, device)
    
    # 2. External JSRT
    jsrt_df = df[(df['split'] == 'Test') & (df['source'] == 'JSRT')]
    jsrt_ds = ChestVisionDataset(jsrt_df, transform=val_transform)
    jsrt_loader = DataLoader(jsrt_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    jsrt_metrics = evaluate_model("External_JSRT", model, jsrt_loader, device)
    
    # Save metrics
    metrics_out = {
        "Internal_Test": test_metrics,
        "External_JSRT": jsrt_metrics
    }
    
    with open(os.path.join(MODEL_STORE, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=4)
        
    logger.info("Evaluation complete. Metrics saved to metrics.json")
    
if __name__ == "__main__":
    train_model()
    run_evaluation()
