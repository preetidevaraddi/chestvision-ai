import os
import pandas as pd
import torch
import json
from sklearn.model_selection import train_test_split
from app.ml.model import ChestVisionModel
from app.core.config import settings

def main():
    print("--- 1. DATASET VERIFICATION ---")
    csv_path = 'dataset/labels.csv'
    df = pd.read_csv(csv_path)
    counts = df['label'].value_counts()
    min_count = counts.min()
    print("Class counts in labels.csv:")
    print(counts)
    print(f"\nMinimum count (used for balancing in train.py): {min_count}")
    print(f"Total balanced dataset size (5 classes): {min_count * 5}")

    print("\n--- 3. DATA LEAKAGE CHECK ---")
    if 'patientid' in df.columns:
        unique_patients = df['patientid'].unique()
        train_p, temp_p = train_test_split(unique_patients, test_size=0.3, random_state=42)
        val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)
        
        train_df = df[df['patientid'].isin(train_p)]
        val_df = df[df['patientid'].isin(val_p)]
        test_df = df[df['patientid'].isin(test_p)]
        
        print("Train patients overlapping with Val:", len(set(train_p).intersection(val_p)))
        print("Train patients overlapping with Test:", len(set(train_p).intersection(test_p)))
        print("Val patients overlapping with Test:", len(set(val_p).intersection(test_p)))
    else:
        print("No patientid column found.")

    print("\n--- 6. CLASS LABEL MAPPING ---")
    with open('models_store/labels.json', 'r') as f:
        saved_labels = json.load(f)
    print(f"Labels in models_store/labels.json: {saved_labels}")
    
    label_cols = sorted(df["label"].unique().tolist())
    print(f"Sorted labels from labels.csv (train.py order): {label_cols}")

    print("\n--- 7. MODEL ARCHITECTURE ---")
    model = ChestVisionModel(num_classes=5)
    
    if os.path.exists(settings.MODEL_PATH):
        try:
            model.load_state_dict(torch.load(settings.MODEL_PATH, map_location='cpu'))
            print("Loaded saved model weights successfully.")
        except Exception as e:
            print(f"Failed to load model weights: {e}")
    else:
        print("Model file not found.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Check what train.py did:
    for name, param in model.backbone.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    
    trainable_params_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params_after_freeze

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters (if frozen as in train.py): {trainable_params_after_freeze}")
    print(f"Frozen parameters: {frozen_params}")

if __name__ == '__main__':
    main()
