import os
import torch
import cv2
import numpy as np
from PIL import Image

from app.ml.infer import get_model, analyze_xray
from app.core.config import settings

def test_visualize():
    import glob
    images = glob.glob("dataset/images/*.png") + glob.glob("dataset/images/*.jpg")
    if not images:
        print("No images found in dataset/images/")
        return
    image_path = images[0]
    
    print(f"Analyzing {image_path}...")
    res = analyze_xray(image_path)
    
    heatmap_url = res.get("heatmap_path")
    print(f"Heatmap URL: {heatmap_url}")
    
    if heatmap_url:
        # /files/uploads/heatmap_xxx.png -> backend/uploads/heatmap_xxx.png
        filename = os.path.basename(heatmap_url)
        local_path = os.path.join(settings.UPLOAD_DIR, filename)
        
        orig_img = cv2.imread(image_path)
        heat_img = cv2.imread(local_path)
        
        diff = cv2.absdiff(orig_img, heat_img)
        print(f"Mean absolute difference: {np.mean(diff):.4f}")
        print(f"Max absolute difference: {np.max(diff)}")
        if np.mean(diff) < 1.0:
            print("ERROR: Images are virtually identical!")
        else:
            print("SUCCESS: Images are different, heatmap is applied.")

if __name__ == "__main__":
    test_visualize()
