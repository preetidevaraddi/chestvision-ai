import pandas as pd
from app.ml.infer import analyze_xray
import os

def main():
    df = pd.read_csv('dataset/labels.csv')
    class_counts = df['label'].value_counts()
    min_count = class_counts.min()

    balanced_dfs = []
    for cls in class_counts.index:
        balanced_dfs.append(df[df['label'] == cls].head(min_count))
    used_df = pd.concat(balanced_dfs)

    unused_df = df[~df['filename'].isin(used_df['filename'])]
    print(f"Total images in CSV: {len(df)}")
    print(f"Used images: {len(used_df)}")
    print(f"Unused images available for testing: {len(unused_df)}")

    for cls in unused_df['label'].unique():
        sample = unused_df[unused_df['label'] == cls].head(2)
        for _, row in sample.iterrows():
            # try multiple paths just in case
            img_path = os.path.join('dataset/images', row['filename'])
            if not os.path.exists(img_path):
                img_path = os.path.join('dataset/real_images', row['filename'])
            
            try:
                res = analyze_xray(img_path)
                print(f"{row['filename']} | true: {row['label']} | pred: {res['top_prediction_label']} | conf: {res['top_prediction_confidence']:.4f}")
            except Exception as e:
                print(f"Failed on {row['filename']}: {e}")

if __name__ == '__main__':
    main()
