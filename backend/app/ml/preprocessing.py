"""
Shared preprocessing pipeline.

CRITICAL: this module is imported by BOTH train.py and infer.py so that
training-time and inference-time preprocessing can never drift apart
(spec section 8).
"""
from PIL import Image
import torch
from torchvision import transforms

from app.core.config import settings


def get_transform(train: bool = False) -> transforms.Compose:
    """
    Returns the exact preprocessing/augmentation pipeline.
    train=True adds light augmentation (used only in train.py);
    the deterministic resize/normalize core is identical in both modes.
    """
    ops = []
    if train:
        ops += [
            transforms.RandomResizedCrop(settings.IMAGE_SIZE, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        ops += [
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        ]
    ops += [
        transforms.Grayscale(num_output_channels=3),  # chest X-rays are grayscale; CNN backbones expect 3ch
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.NORMALIZE_MEAN, std=settings.NORMALIZE_STD),
    ]
    return transforms.Compose(ops)


def validate_image(file_path: str) -> bool:
    """Confirms the file is a readable, non-corrupt image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_and_preprocess(file_path: str, train: bool = False) -> torch.Tensor:
    """Load an image from disk and return a preprocessed tensor ready for the model."""
    image = Image.open(file_path).convert("RGB")
    transform = get_transform(train=train)
    return transform(image)
