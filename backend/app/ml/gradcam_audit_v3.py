"""
V3 Grad-CAM Audit Script
Generates Grad-CAM visualizations for 1 representative sample from each of the 5 canonical classes.
"""
import os
import json
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from app.ml.model import ChestVisionModel
from app.ml.gradcam import GradCAM, save_overlay
from app.core.config import settings

def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.NORMALIZE_MEAN, std=settings.NORMALIZE_STD),
    ])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Grad-CAM Audit on {device}...")
    
    V3_DIR = r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\models_store\new_model_v3"
    MANIFEST_PATH = r"C:\Users\Preeti Devaraddi\Documents\chestvision-ai-copy\backend\dataset\v3_workspace\dataset_manifest.csv"
    
    with open(os.path.join(V3_DIR, "labels.json"), "r") as f:
        label_cols = json.load(f)
    label_map = {lbl: idx for idx, lbl in enumerate(label_cols)}
    
    model = ChestVisionModel(num_classes=len(label_cols), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(V3_DIR, "best_model.pt"), map_location=device))
    model = model.to(device)
    model.eval()
    
    # We use the final dense block norm5 layer for DenseNet121 Grad-CAM
    target_layer = model.backbone.features.norm5
    gradcam = GradCAM(model, target_layer)
    
    transform = get_eval_transform()
    
    df = pd.read_csv(MANIFEST_PATH)
    test_df = df[df['split'] == 'Test']
    
    # Pick 1 representative image per class
    os.makedirs(os.path.join(V3_DIR, "gradcam_audit"), exist_ok=True)
    
    audit_results = []
    for cls in label_cols:
        sample = test_df[test_df['label'] == cls].head(1)
        if sample.empty:
            continue
        
        row = sample.iloc[0]
        img_path = str(row['filepath'])
        cls_idx = label_map[cls]
        
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        heatmap = gradcam.generate(input_tensor, cls_idx)
        out_path = os.path.join(V3_DIR, "gradcam_audit", f"gradcam_{cls}.png")
        save_overlay(img_path, heatmap, out_path)
        
        print(f"Generated Grad-CAM for {cls}: {out_path}")
        audit_results.append({
            "class": cls,
            "source_file": img_path,
            "heatmap_file": out_path
        })
        
    with open(os.path.join(V3_DIR, "gradcam_audit", "audit_log.json"), "w") as f:
        json.dump(audit_results, f, indent=2)
        
    print("\nGrad-CAM Audit Complete!")

if __name__ == "__main__":
    main()
