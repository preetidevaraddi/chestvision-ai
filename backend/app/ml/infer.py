"""
Inference service.

Loads the trained model ONCE at process startup (spec section 29: the app
must NOT retrain on every upload) and exposes a single `analyze_xray()`
entry point that runs the full pipeline:

Preprocessing -> Feature extraction (backbone) -> Disease classification
-> Confidence/uncertainty -> Suspicious-region heatmap -> Priority

IMPORTANT: this module keeps two clearly separate model states:
  - TRAINED model  (settings.MODEL_PATH exists, produced by train.py on a
    real dataset) -> model_version = "trained"
  - DEMO FALLBACK model (no trained weights yet - either an ImageNet-pretrained
    backbone, or a randomly-initialized one if weights can't be downloaded)
    -> model_version = "demo_fallback"

Every result carries is_demo_model + a disclaimer string so the frontend/report
can never present a demo-fallback prediction as clinically meaningful.
"""
import os
import uuid
import torch

from app.core.config import settings, get_disease_labels
from app.ml.preprocessing import load_and_preprocess
from app.ml.model import build_model, load_trained_model
from app.ml.gradcam import GradCAM, save_overlay
from app.ml.analysis_rules import classify_uncertainty, assess_priority

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_labels = None
_is_demo = None

DEMO_DISCLAIMER = (
    "DEMO MODE: no trained model is loaded. This prediction comes from an "
    "untrained/ImageNet-pretrained backbone and carries no diagnostic or "
    "academic evaluation value. Run app/ml/train.py on a labeled chest "
    "X-ray dataset to produce a real model."
)
TRAINED_DISCLAIMER = (
    "This is an academic final-year project prototype. The AI model is not "
    "clinically validated and must not be used for real medical diagnosis."
)


def get_model():
    """Loads the model once and reuses it. Returns (model, labels, is_demo)."""
    global _model, _labels, _is_demo
    if _model is None:
        _labels = get_disease_labels()
        if os.path.exists(settings.MODEL_PATH):
            _model = load_trained_model(settings.MODEL_PATH, num_classes=len(_labels), device=_device)
            _is_demo = False
        else:
            # DEMO FALLBACK: no trained weights yet. Always a RANDOMLY
            # INITIALIZED backbone (no ImageNet weight download - keeps this
            # mode fast, offline-safe, and unambiguous: it is a wiring
            # demonstration only, never dressed up as a "pretrained model").
            _model = build_model(num_classes=len(_labels), pretrained=False).to(_device)
            _model.eval()
            _is_demo = True
    return _model, _labels, _is_demo


def reset_model_cache():
    """Call after training completes so the API picks up the newly trained
    model instead of continuing to serve the demo fallback / a stale model."""
    global _model, _labels, _is_demo
    _model, _labels, _is_demo = None, None, None


def analyze_xray(image_path: str) -> dict:
    model, labels, is_demo = get_model()

    input_tensor = load_and_preprocess(image_path, train=False).unsqueeze(0).to(_device)
    input_tensor.requires_grad_()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_conditions = sorted(
        [{"label": label, "confidence": float(prob)} for label, prob in zip(labels, probs)],
        key=lambda x: x["confidence"],
        reverse=True,
    )
    top = predicted_conditions[0]
    uncertainty_status = classify_uncertainty(top["confidence"])
    priority = assess_priority(predicted_conditions, top["label"], top["confidence"])

    # Suspicious-region visualization (Grad-CAM) for the top predicted class
    heatmap_url = None
    try:
        cam = GradCAM(model, model.get_last_conv_layer())
        top_idx = labels.index(top["label"])
        heatmap = cam.generate(input_tensor, top_idx)
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.png"
        absolute_heatmap_path = os.path.join(settings.UPLOAD_DIR, heatmap_filename)
        save_overlay(image_path, heatmap, absolute_heatmap_path)
        heatmap_url = f"/files/uploads/{heatmap_filename}"
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Grad-CAM generation failed: {e}")
        heatmap_url = None  # explanation is best-effort; never block the core result

    return {
        "predicted_conditions": predicted_conditions,
        "top_prediction_label": top["label"],
        "top_prediction_confidence": top["confidence"],
        "uncertainty_status": uncertainty_status,
        "priority": priority,
        "heatmap_path": heatmap_url,
        "model_version": "demo_fallback" if is_demo else "trained",
        "is_demo_model": is_demo,
        "disclaimer": DEMO_DISCLAIMER if is_demo else TRAINED_DISCLAIMER,
    }
