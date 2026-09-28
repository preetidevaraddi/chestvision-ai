import os
import json
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from app.ml.infer import analyze_xray
from app.ml.preprocessing import load_and_preprocess

def main():
    print("=== MODEL EVALUATION ===")
    df = pd.read_csv('dataset/labels_cleaned.csv')
    
    # Imitate split logic from final_audit.py
    min_count = df['label'].value_counts().min()
    balanced_dfs = [df[df['label'] == cls].head(min_count) for cls in df['label'].unique()]
    bdf = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    used_filenames = bdf['filename'].tolist()
    unused_df = df[~df['filename'].isin(used_filenames)].copy()
    
    # Make sure we have a decent number of unused images per class
    # To not take forever, we'll cap it at 100 images per class for evaluation, or use all if less.
    eval_df = pd.DataFrame()
    for cls in unused_df['label'].unique():
        cls_df = unused_df[unused_df['label'] == cls]
        # cap at 100 for evaluation speed, or maybe 50
        eval_df = pd.concat([eval_df, cls_df.head(200)]) 
        
    print(f"Total used in train/val/test: {len(bdf)}")
    print(f"Total unseen available: {len(unused_df)}")
    print(f"Total used for unseen evaluation: {len(eval_df)}")
    
    y_true = []
    y_pred = []
    failures = []
    
    for _, row in eval_df.iterrows():
        img_path = os.path.join('dataset/images', row['filename'])
        if not os.path.exists(img_path):
            img_path = os.path.join('dataset/real_images', row['filename'])
            if not os.path.exists(img_path):
                continue
                
        try:
            res = analyze_xray(img_path)
            true_label = row['label']
            pred_label = res['top_prediction_label']
            conf = res['top_prediction_confidence']
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            if true_label != pred_label:
                failures.append({
                    'filename': row['filename'],
                    'true_class': true_label,
                    'predicted_class': pred_label,
                    'confidence': conf,
                    'source': row.get('source', 'unknown')
                })
        except Exception as e:
            print(f"Error analyzing {row['filename']}: {e}")
            
    # Calculate metrics
    labels = sorted(list(set(y_true + y_pred)))
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_per, rec_per, f1_per, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    results = {
        'accuracy': acc,
        'macro_precision': prec_macro,
        'macro_recall': rec_macro,
        'macro_f1': f1_macro,
        'per_class': {
            label: {
                'precision': prec_per[i],
                'recall': rec_per[i],
                'f1': f1_per[i]
            } for i, label in enumerate(labels)
        },
        'confusion_matrix': cm.tolist(),
        'cm_labels': labels,
        'failures': failures
    }
    
    with open('eval_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Evaluation completed. Results saved to eval_results.json.")

if __name__ == '__main__':
    main()
