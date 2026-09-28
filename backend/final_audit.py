import os
import hashlib
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from app.ml.model import ChestVisionModel
from app.ml.preprocessing import load_and_preprocess
from app.ml.gradcam import GradCAM, save_overlay
from app.ml.infer import analyze_xray
import uuid

def get_hash(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    print("=== 1. DATASET / DUPLICATES ===")
    df = pd.read_csv('dataset/labels_cleaned.csv')
    print("Final class counts:")
    print(df['label'].value_counts())
    
    hashes = {}
    duplicates = 0
    for _, row in df.iterrows():
        img_path = os.path.join('dataset/images', row['filename'])
        if not os.path.exists(img_path):
            img_path = os.path.join('dataset/real_images', row['filename'])
            if not os.path.exists(img_path):
                continue
        h = get_hash(img_path)
        if h in hashes:
            duplicates += 1
        hashes[h] = True
    print(f"Exact duplicates remaining in cleaned dataset: {duplicates}")

    print("\n=== 2. DATA SPLIT ===")
    # Re-run split logic to verify counts
    class_counts = df['label'].value_counts()
    min_count = class_counts.min()
    balanced_dfs = [df[df['label'] == cls].head(min_count) for cls in class_counts.index]
    bdf = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    if 'patientid' in bdf.columns:
        print("Using patient-aware split.")
        unique_patients = bdf['patientid'].unique()
        train_p, temp_p = train_test_split(unique_patients, test_size=0.3, random_state=42)
        val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=42)
        
        train_df = bdf[bdf['patientid'].isin(train_p)]
        val_df = bdf[bdf['patientid'].isin(val_p)]
        test_df = bdf[bdf['patientid'].isin(test_p)]
        
        print(f"Train unique patients: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")
        print(f"Train/Val patient overlap: {len(set(train_p).intersection(val_p))}")
        print(f"Train/Test patient overlap: {len(set(train_p).intersection(test_p))}")
        print(f"Val/Test patient overlap: {len(set(val_p).intersection(test_p))}")
    
    print("\nTrain per-class counts:")
    print(train_df['label'].value_counts())
    print("\nVal per-class counts:")
    print(val_df['label'].value_counts())
    print("\nTest per-class counts:")
    print(test_df['label'].value_counts())
    
    print("\n=== 3. MODEL ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_path = "models_store/new_model_v2/labels.json"
    model_path = "models_store/new_model_v2/chestvision_model.pt"
    
    with open(labels_path, "r") as f:
        labels = json.load(f)
    print(f"Labels loaded: {labels}")
    
    model = ChestVisionModel(num_classes=len(labels), pretrained=False)
    # Load state dict
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Model loaded successfully in fresh process.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        
    total_params = sum(p.numel() for p in model.parameters())
    # Note: when we load weights, everything gets its default requires_grad (True) unless we explicitly recreate the frozen state.
    # We can check what train_v2.py did:
    trainable_params_in_train = 0
    for name, param in model.backbone.named_parameters():
        if "denseblock4" in name or "norm5" in name or "classifier" in name:
            trainable_params_in_train += param.numel()
    print(f"Total params: {total_params}")
    print(f"Trainable params (as configured in train_v2): {trainable_params_in_train}")
    
    print("\n=== 6. GENERALIZATION ===")
    unused_df = df[~df['filename'].isin(bdf['filename'])]
    model.eval()
    
    for cls in unused_df['label'].unique():
        sample = unused_df[unused_df['label'] == cls].head(2)
        for _, row in sample.iterrows():
            img_path = os.path.join('dataset/images', row['filename'])
            if not os.path.exists(img_path):
                img_path = os.path.join('dataset/real_images', row['filename'])
                if not os.path.exists(img_path): continue
            
            input_tensor = load_and_preprocess(img_path, train=False).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(model(input_tensor), dim=1)[0].cpu().numpy()
            
            pred_idx = probs.argmax()
            pred_lbl = labels[pred_idx]
            correct = "Correct" if pred_lbl == row['label'] else "Incorrect"
            print(f"{row['filename']} | true: {row['label']} | pred: {pred_lbl} | conf: {probs[pred_idx]:.4f} | {correct}")
            
    print("\n=== 7. GRAD-CAM / AFFECTED AREA ===")
    # Pick one test image
    test_img = unused_df.iloc[0]
    test_img_path = os.path.join('dataset/images', test_img['filename'])
    if not os.path.exists(test_img_path):
        test_img_path = os.path.join('dataset/real_images', test_img['filename'])
        
    input_tensor = load_and_preprocess(test_img_path, train=False).unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    
    try:
        cam = GradCAM(model, model.get_last_conv_layer())
        # Predict first
        logits = model(input_tensor)
        top_idx = logits.argmax().item()
        
        heatmap = cam.generate(input_tensor, top_idx)
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        heatmap_path = os.path.join('uploads', heatmap_filename) # Using uploads dir as in infer.py
        
        save_overlay(test_img_path, heatmap, heatmap_path)
        
        if os.path.exists(heatmap_path) and os.path.getsize(heatmap_path) > 0:
            print(f"Grad-CAM Heatmap generated successfully: {heatmap_path}")
            print(f"File size: {os.path.getsize(heatmap_path)} bytes")
        else:
            print("Heatmap generation failed (file empty or missing).")
    except Exception as e:
        print(f"Grad-CAM Error: {e}")

if __name__ == '__main__':
    main()
