import os
import pandas as pd
from app.ml.infer import analyze_xray

def main():
    df = pd.read_csv('dataset/labels_cleaned.csv')
    img_row = df.iloc[0]
    img_path = os.path.join('dataset/images', img_row['filename'])
    if not os.path.exists(img_path):
        img_path = os.path.join('dataset/real_images', img_row['filename'])
    
    print(f"Testing analyze_xray on {img_path}")
    result = analyze_xray(img_path)
    
    print(f"Top Pred: {result['top_prediction_label']}")
    print(f"Heatmap path: {result['heatmap_path']}")
    
    if result['heatmap_path'] is None:
        print("HEATMAP WAS NONE! AN EXCEPTION PROBABLY OCCURRED!")
    else:
        print("Heatmap generated successfully.")
        
if __name__ == '__main__':
    main()
