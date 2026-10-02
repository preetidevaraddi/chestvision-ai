"""
Grad-CAM: produces a heatmap over the regions of the X-ray that most
influenced the model's prediction. This is a visual EXPLANATION, not a
diagnostic localization - the report wording must reflect that (spec §15).
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from app.core.config import settings

def create_lung_mask(img_gray: np.ndarray) -> np.ndarray:
    """Create a lightweight lung-region mask using Otsu thresholding and contour filtering."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img_gray)
    
    blur = cv2.GaussianBlur(enhanced, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(img_gray)
    h, w = img_gray.shape
    
    for cnt in contours:
        x, y, w_bb, h_bb = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        margin = int(w * 0.02)
        touching_edge = (x <= margin) or (y <= margin) or ((x + w_bb) >= w - margin) or ((y + h_bb) >= h - margin)
        
        if area > (h * w) * 0.015 and not touching_edge:
            cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)
            
    if np.sum(mask) == 0:
        # Fallback to a central bounding box if heuristics fail
        cv2.rectangle(mask, (int(w*0.15), int(h*0.15)), (int(w*0.85), int(h*0.85)), 1, thickness=cv2.FILLED)
        
    # Smooth the mask for a natural blend
    mask = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
    mask = np.clip(mask, 0, 1)
    
    return mask

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # 4. Gradient pooling: global average pool over spatial dimensions (H, W) -> shape (1, C, 1, 1)
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 5. Weighted activation combination
        activations = self.activations * pooled_gradients
        
        # Sum across channels
        heatmap = torch.sum(activations, dim=1).squeeze(0).cpu().numpy()
        
        # 6. ReLU application (discard negative activations)
        heatmap = np.maximum(heatmap, 0)
        
        # --- Remove Edge/Corner Artifacts ---
        # CNNs often produce high gradients at the image borders due to padding.
        # Zeroing out the 1-pixel border of the low-res feature map eliminates these before normalization.
        if heatmap.shape[0] > 2 and heatmap.shape[1] > 2:
            heatmap[0, :] = 0
            heatmap[-1, :] = 0
            heatmap[:, 0] = 0
            heatmap[:, -1] = 0
            
        # 7. Heatmap normalization to [0, 1]
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap /= heatmap_max
            
        return heatmap

def save_overlay(original_image_path: str, heatmap: np.ndarray, output_path: str) -> str:
    """Overlays the heatmap on the original X-ray and saves it for report/UI display."""
    img = Image.open(original_image_path).convert("RGB")
    original_w, original_h = img.size
    img_arr = np.array(img)

    # 9. Resizing/interpolation (using INTER_LINEAR or INTER_CUBIC)
    heatmap_resized = cv2.resize(heatmap, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    # 10. Apply Lung Mask to strictly restrict heatmap to anatomical lungs
    gray_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    lung_mask = create_lung_mask(gray_img)
    heatmap_cleaned = heatmap_resized * lung_mask

    # Re-normalize after lung masking to ensure highest activation is bright red
    max_val = np.max(heatmap_cleaned)
    if max_val > 0:
        heatmap_cleaned = heatmap_cleaned / max_val

    # 11. Final overlay with original X-ray
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_cleaned), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Alpha masking: opacity is proportional to activation intensity.
    # Set a maximum opacity so it doesn't completely block the X-ray
    alpha_mask = heatmap_cleaned[..., np.newaxis] * 0.55
    
    # Where heatmap is 0, alpha is 0, so the original X-ray is preserved perfectly.
    overlay = np.uint8(img_arr * (1 - alpha_mask) + heatmap_color * alpha_mask)
    
    Image.fromarray(overlay).save(output_path)
    return output_path
