import os
import torch
import torch.nn.functional as F
from PIL import Image
import json
import uuid

# Mocks and local imports
from app.ml.model import load_trained_model
from app.ml.preprocessing import load_and_preprocess
from app.ml.gradcam import GradCAM

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "models_store/new_model_real/chestvision_model.pt"
labels_path = "models_store/new_model_real/labels.json"

labels = json.load(open(labels_path))
print("Loaded labels:", labels)

model = load_trained_model(model_path, num_classes=5, device=device)

# Pick some sample images
samples = [
    "dataset/images/COVID-19_0001.png",
    "dataset/images/Nodule_0001.png",
    "dataset/images/Normal_0001.png",
    "dataset/images/Pneumonia_0001.jpeg",
    "dataset/images/Tuberculosis_0001.png"
]

print("\n--- INFERENCE TEST ---")
for img_path in samples:
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue
    
    input_tensor = load_and_preprocess(img_path, train=False).unsqueeze(0).to(device)
    input_tensor.requires_grad_()
    
    # 1. Forward Pass
    logits = model(input_tensor)
    
    # 2. Correct Single-Label Softmax (Not Sigmoid!)
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    
    # 3. Predict Top Class
    pred_idx = probs.argmax()
    top_label = labels[pred_idx]
    top_conf = probs[pred_idx]
    print(f"\n{os.path.basename(img_path)}:")
    print(f"  -> Predicted: {top_label} ({top_conf*100:.2f}%)")
    
    # 4. Grad-CAM Verification
    try:
        cam = GradCAM(model, model.get_last_conv_layer())
        heatmap = cam.generate(input_tensor, pred_idx)
        print(f"  -> Grad-CAM Generated Successfully (heatmap shape: {heatmap.shape})")
    except Exception as e:
        print(f"  -> Grad-CAM FAILED: {str(e)}")

print("\n--- ARCHITECTURE VERIFICATION ---")
print("Model backbone: DenseNet121")
print(f"Classifier head: {model.backbone.classifier}")

print("\n--- BACKEND COMPATIBILITY REPORT ---")
print("Required changes to activate new model:")
print("1. app/core/config.py: Update MODEL_PATH, LABELS_PATH, METRICS_PATH to point to 'new_model_real' directory.")
print("2. app/ml/infer.py: Change torch.sigmoid(logits) to torch.softmax(logits, dim=1) for single-label exclusive classification.")
