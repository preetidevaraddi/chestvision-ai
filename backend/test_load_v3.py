import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.ml.model import load_trained_model

MODEL_PATH = "models_store/new_model_v3/chestvision_model.pt"

try:
    model = load_trained_model(MODEL_PATH, num_classes=5, device="cpu")
    print("SUCCESS: Model loaded successfully.")
    
    # Test inference
    test_tensor = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(test_tensor)
    
    print(f"Output shape: {out.shape}")
    if out.shape == (1, 5):
        print("SUCCESS: Model is compatible with 5-class inference.")
    else:
        print("FAILED: Output shape mismatch.")
except Exception as e:
    print(f"FAILED to load model: {e}")
