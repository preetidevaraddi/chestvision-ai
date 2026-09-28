"""
Central configuration for ChestVision AI.
Everything the spec (section 30) requires to be configurable lives here,
overridable via environment variables / a .env file.
"""
import os
import json
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/


class Settings(BaseSettings):
    # --- App ---
    APP_NAME: str = "ChestVision AI"
    SECRET_KEY: str = "change-this-secret-key-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8  # 8 hours

    # --- Database (MySQL via SQLAlchemy + PyMySQL driver) ---
    DATABASE_URL: str = "mysql+pymysql://chestvision_user:ChestVision%40123@localhost:3306/chestvision"

    # --- Storage paths ---
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    REPORT_DIR: str = str(BASE_DIR / "reports")
    MODEL_DIR: str = str(BASE_DIR / "models_store")
    DATASET_DIR: str = str(BASE_DIR / "dataset")
    MODEL_PATH: str = str(BASE_DIR / "models_store" / "new_model_v3_extended" / "best_model.pt")
    LABELS_PATH: str = str(BASE_DIR / "models_store" / "new_model_v3_extended" / "labels.json")
    METRICS_PATH: str = str(BASE_DIR / "models_store" / "new_model_v3_extended" / "metrics.json")

    # --- Upload limits ---
    MAX_UPLOAD_SIZE_MB: int = 15
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]

    # --- Image preprocessing (must match exactly between train.py and inference) ---
    IMAGE_SIZE: int = 224
    NORMALIZE_MEAN: List[float] = [0.485, 0.456, 0.406]
    NORMALIZE_STD: List[float] = [0.229, 0.224, 0.225]

    # --- Confidence / uncertainty thresholds (spec section 14) ---
    HIGH_CONFIDENCE_THRESHOLD: float = 0.75
    MODERATE_CONFIDENCE_THRESHOLD: float = 0.50
    # below MODERATE_CONFIDENCE_THRESHOLD -> "Requires Review"

    # --- Priority rules (spec section 16) ---
    URGENT_CONFIDENCE_THRESHOLD: float = 0.85
    HIGH_PRIORITY_CONFIDENCE_THRESHOLD: float = 0.60
    SEVERE_CONDITIONS: List[str] = ["Pneumothorax", "Pneumonia", "Mass", "Consolidation"]

    class Config:
        env_file = ".env"


settings = Settings()

for d in [settings.UPLOAD_DIR, settings.REPORT_DIR, settings.MODEL_DIR, settings.DATASET_DIR]:
    os.makedirs(d, exist_ok=True)


def get_disease_labels() -> List[str]:
    """
    Disease classes are configurable, derived from the dataset/config used at
    training time - never hard-coded into model logic (spec section 9).
    Falls back to a default demo label set (NIH ChestX-ray14 style) only
    until a model has actually been trained.
    """
    if os.path.exists(settings.LABELS_PATH):
        with open(settings.LABELS_PATH, "r") as f:
            return json.load(f)
    default_labels = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia",
    ]
    with open(settings.LABELS_PATH, "w") as f:
        json.dump(default_labels, f)
    return default_labels
