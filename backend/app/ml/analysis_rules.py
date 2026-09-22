"""
Confidence/uncertainty classification and priority/urgency assessment.
Pure, testable functions driven entirely by configured thresholds
(spec section 30) - no hard-coded disease-specific branches.
"""
from typing import List, Dict
from app.core.config import settings


def classify_uncertainty(top_confidence: float) -> str:
    if top_confidence >= settings.HIGH_CONFIDENCE_THRESHOLD:
        return "High Confidence"
    if top_confidence >= settings.MODERATE_CONFIDENCE_THRESHOLD:
        return "Moderate Confidence"
    return "Requires Review"


def assess_priority(predicted_conditions: List[Dict], top_label: str, top_confidence: float) -> str:
    """
    AI-ASSISTED prioritization only - explicitly NOT a diagnosis.
    Rules:
      - Urgent: very high confidence on a configured severe condition
      - High:   moderate-high confidence on a severe condition, OR very high
                confidence on any non-"No Finding" condition
      - Normal: everything else (including confident "No Finding")
    """
    if top_label == "No Finding" or top_label.lower() == "normal":
        return "Normal"

    is_severe = top_label in settings.SEVERE_CONDITIONS

    if is_severe and top_confidence >= settings.URGENT_CONFIDENCE_THRESHOLD:
        return "Urgent"
    if (is_severe and top_confidence >= settings.HIGH_PRIORITY_CONFIDENCE_THRESHOLD) or (
        top_confidence >= settings.URGENT_CONFIDENCE_THRESHOLD
    ):
        return "High"
    return "Normal"
