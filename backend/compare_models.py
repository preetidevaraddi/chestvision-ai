import json
import os
import pandas as pd
from app.ml.model import ChestVisionModel
from app.ml.preprocessing import load_and_preprocess
import torch

def test_unseen_new_model_fixed():
    print(f"\n{'='*15} UNSEEN DATA TEST (NEW MODEL) {'='*15}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    labels_path = "models_store/new_model_v2/labels.json"
    model_path = "models_store/new_model_v2/chestvision_model.pt"
    
    with open(labels_path, "r") as f:
        labels = json.load(f)
        
    model = ChestVisionModel(num_classes=len(labels), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    df_clean = pd.read_csv('dataset/labels_cleaned.csv')
    min_count = df_clean['label'].value_counts().min()
    
    balanced_dfs = []
    for cls in df_clean['label'].value_counts().index:
        balanced_dfs.append(df_clean[df_clean['label'] == cls].head(min_count))
    used_df = pd.concat(balanced_dfs)
    
    # Leftovers from the CLEANED dataset (no MD5 duplicates)
    unused_df = df_clean[~df_clean['filename'].isin(used_df['filename'])]
    
    for cls in unused_df['label'].unique():
        sample = unused_df[unused_df['label'] == cls].head(2)
        for _, row in sample.iterrows():
            img_path = os.path.join('dataset/images', row['filename'])
            if not os.path.exists(img_path):
                img_path = os.path.join('dataset/real_images', row['filename'])
                if not os.path.exists(img_path):
                    continue
            
            input_tensor = load_and_preprocess(img_path, train=False).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                
            pred_idx = probs.argmax()
            pred_lbl = labels[pred_idx]
            conf = probs[pred_idx]
            
            print(f"{row['filename']:20} | true: {row['label']:12} | pred: {pred_lbl:12} | conf: {conf:.4f}")

if __name__ == '__main__':
    test_unseen_new_model_fixed()
