import os
import sys
import torch

# Ensure we can import from app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ml.infer import get_model
from app.ml.preprocessing import load_and_preprocess
from app.ml.analysis_rules import classify_uncertainty, assess_priority

def run_diagnostics(image_path):
    print(f"Image: {image_path}")
    model, labels, is_demo = get_model()
    print(f"Model checkpoint: {'demo_fallback' if is_demo else 'trained'}")
    
    # Preprocessing
    print(f"Preprocessing: Using load_and_preprocess from app.ml.preprocessing")
    input_tensor = load_and_preprocess(image_path, train=False).unsqueeze(0).to(next(model.parameters()).device)
    input_tensor.requires_grad_()
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
        
    print(f"Raw logits: {logits.cpu().numpy().tolist()[0]}")
    print(f"Softmax probabilities for all classes: {probs.tolist()}")
    print(f"Labels: {labels}")
    print(f"Number of logits: {len(logits[0])}, Number of labels: {len(labels)}")
    
    predicted_conditions = sorted(
        [{"label": label, "confidence": float(prob)} for label, prob in zip(labels, probs)],
        key=lambda x: x["confidence"],
        reverse=True,
    )
    top = predicted_conditions[0]
    
    # Wait, infer.py does this:
    # top["confidence"] is directly float(prob). Let's see if it's being returned.
    print(f"Predicted class: {top['label']}")
    print(f"Confidence: {top['confidence']}")
    
    priority = assess_priority(predicted_conditions, top["label"], top["confidence"])
    print(f"Priority: {priority}")
    
    print(f"Grad-CAM target class: {top['label']} (Index: {labels.index(top['label'])})")
    
    from app.ml.gradcam import GradCAM, save_overlay
    from app.core.config import settings
    import uuid
    heatmap_generated = False
    try:
        cam = GradCAM(model, model.get_last_conv_layer())
        top_idx = labels.index(top["label"])
        heatmap = cam.generate(input_tensor, top_idx)
        heatmap_generated = True
    except Exception as e:
        pass
    print(f"Heatmap generated: {heatmap_generated}")
    
    print(f"Active checkpoint path: {settings.MODEL_PATH}")

if __name__ == "__main__":
    test_image = "test_n5.png" # There is a test image in backend/ directory
    if os.path.exists(test_image):
        run_diagnostics(test_image)
    else:
        print("Test image not found.")
