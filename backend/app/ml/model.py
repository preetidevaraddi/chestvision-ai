"""
Model definition.

Uses a torchvision DenseNet-121 backbone (the standard architecture used in
chest X-ray literature) with a linear layer sized to the
configured mutually exclusive disease labels, for SINGLE-LABEL classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


class ChestVisionModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        # Manually replicate torchvision's DenseNet.forward but WITHOUT the
        # in-place ReLU it normally uses. The in-place op corrupts the
        # activation tensor that Grad-CAM's forward hook captures, which
        # breaks the backward pass used for suspicious-region heatmaps.
        features = self.backbone.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.backbone.classifier(out)
        return out

    def get_last_conv_layer(self):
        # used by Grad-CAM for suspicious-region visualization
        # In DenseNet121, denseblock4 is 7x7, which produces an overly broad/diffuse heatmap when upscaled.
        # denseblock3 is 14x14, providing 4x higher spatial resolution for tighter, more accurate clinical localization.
        return self.backbone.features.denseblock3


def build_model(num_classes: int, pretrained: bool = True) -> ChestVisionModel:
    return ChestVisionModel(num_classes=num_classes, pretrained=pretrained)


def load_trained_model(model_path: str, num_classes: int, device: str = "cpu") -> ChestVisionModel:
    model = build_model(num_classes=num_classes, pretrained=False)
    state = torch.load(model_path, map_location=device, weights_only=True)
    
    # Handle both wrapped and unwrapped state dicts
    mapped_state = {}
    for k, v in state.items():
        if not k.startswith('backbone.'):
            mapped_state[f'backbone.{k}'] = v
        else:
            mapped_state[k] = v
            
    model.load_state_dict(mapped_state)
    model.to(device)
    model.eval()
    return model
