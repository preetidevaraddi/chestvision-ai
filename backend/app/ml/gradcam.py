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
            
        # 8. Heatmap sharpening & thresholding
        heatmap = heatmap ** 2
        
        # Threshold to remove weak activations
        heatmap[heatmap < 0.3] = 0
            
        return heatmap

def save_overlay(original_image_path: str, heatmap: np.ndarray, output_path: str) -> str:
    """Overlays the heatmap on the original X-ray and saves it for report/UI display."""
    img = Image.open(original_image_path).convert("RGB")
    original_w, original_h = img.size
    img_arr = np.array(img)

    # 9. Resizing/interpolation (using INTER_LINEAR or INTER_CUBIC)
    # Clip to [0, 1] after resize to handle overshoot from cubic interpolation
    heatmap_resized = cv2.resize(heatmap, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    # --- CONNECTED COMPONENT ANALYSIS & FILTERING ---
    # Create a binary mask to identify high-activation regions
    binary_mask = (heatmap_resized > 0.1).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Filter out small/noisy regions (e.g., ignore regions with very small area)
    min_area = (original_w * original_h) * 0.015  # 1.5% of total area
    valid_contours = []
    clean_mask = np.zeros((original_h, original_w), dtype=np.uint8)
    
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            valid_contours.append(contour)
            cv2.drawContours(clean_mask, [contour], -1, 1, thickness=cv2.FILLED)
            
    # Apply the mask to eliminate random spots in background/corners
    heatmap_cleaned = heatmap_resized * clean_mask
    
    # Re-normalize so the valid regions represent full high activation (red/yellow)
    max_val = np.max(heatmap_cleaned)
    if max_val > 0:
        heatmap_cleaned = heatmap_cleaned / max_val

    # 10. Final overlay with original X-ray
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_cleaned), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Alpha masking: opacity is proportional to activation intensity.
    alpha_mask = heatmap_cleaned[..., np.newaxis] * 0.50
    overlay = np.uint8(img_arr * (1 - alpha_mask) + heatmap_color * alpha_mask)
    
    # --- ADD DASHED BLACK OUTLINE ---
    dash_length = 15
    gap_length = 15
    thickness = 3
    
    for contour in valid_contours:
        N = len(contour)
        for i in range(0, N, dash_length + gap_length):
            end = min(i + dash_length, N)
            segment = contour[i:end]
            if len(segment) > 1:
                cv2.polylines(overlay, [segment], isClosed=False, color=(0, 0, 0), thickness=thickness)
    
    Image.fromarray(overlay).save(output_path)
    return output_path
